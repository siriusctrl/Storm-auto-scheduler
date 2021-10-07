import numpy as np
from numpy.lib.polynomial import polyint
import torch
import gym
import argparse
import os

import sys
sys.path.append(os.path.join(os.getcwd(), 'Simulator'))

import pickle

from WordCounting import WordCountingEnv as wc_large
from WordCounting_small import WordCountingEnv as wc_small
from complex_log import ComplexLogEnv as complex_log
from ParallelWrapper import ParalllelWrapper

from SimpleBuffer import ReplayBuffer
from ourDDPG import DDPG
from TD3 import TD3
from Wolptinger.ContinuousCartPole import ContinuousCartPoleEnv


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, eval_episodes=1):
    avg_reward = 0.
    step_count = 0
    res = {}
    for _ in range(eval_episodes):
        state, done = eval_env.reset()[0], False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            step_count += 1
            if step_count >= 10:
                break

            for key in info.keys():
                res[key] = res.get(key, []) + [info[key]]

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} steps:{avg_reward/step_count:.3f}")
    print("---------------------------------------")
    return res


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=10, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=3000, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name              
    parser.add_argument("--n_env", default=10, type=int)
    parser.add_argument("--extra_name", default="")       
    parser.add_argument("--reschedule_cost", default=False, type=bool)
    parser.add_argument("--env_size", default='samll')
    parser.add_argument("--reschedule_weights", default=1, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    args = parser.parse_args()

    file_name = f".{args.env_size}_{args.extra_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: cSim, Seed: {args.seed}, offline parallel condense")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if args.env_size == 'small':
        parallel_env = ParalllelWrapper([wc_small(seed=args.seed) for _ in range(args.n_env)], args.n_env)
        eval_env = wc_small(seed=args.seed+100)
    elif args.env_size == 'large':
        parallel_env = ParalllelWrapper([wc_large(seed=args.seed) for _ in range(args.n_env)], args.n_env)
        eval_env = wc_large(seed=args.seed+100)
    elif args.env_size == 'complex':
        parallel_env = ParalllelWrapper([complex_log(seed=args.seed) for _ in range(args.n_env)], args.n_env)
        eval_env = complex_log(seed=args.seed+100)
    else:
        raise ValueError(f'Unknown env size {args.env_size}')

    with open(f'/offline/random_{file_name}.pkl', 'rb') as f:
        replay_buffer = pickle.load(f)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    min_action = float(eval_env.action_space.low[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    
    if args.policy == "DDPG":
        policy = DDPG(**kwargs)
    elif args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)
        policy.min_action = min_action
    else:
        raise NotImplementedError
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    

    evaluations = []
    states, done = parallel_env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_step_collection = {'pre':[[] for _ in range(args.n_env)]}

    for t in range(int(args.max_timesteps // 100)):
        episode_timesteps += 1

        # perform the offline training
        for _ in range(100):
            policy.train(replay_buffer, args.batch_size)

        states, done = parallel_env.reset(), False
        actions = []
        for state in states:
            actions.append((
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(min_action, max_action))
        
        # next_states, reward, done, info = parallel_env.step_multiple(actions)
        res = parallel_env.step_multiple(actions)
        next_states = []
        batch_reward = 0
        for index in range(len(res)):
            next_state, reward, done, info = res[index]
            next_states.append(next_state)
            batch_reward += reward
            for key in info.keys():
                if key in total_step_collection:
                    total_step_collection[key].append(info[key])
                else:
                    total_step_collection[key] = [info[key]]

        print(f'batch:{t+1}/{int(args.max_timesteps // 100)} avg reward is {batch_reward/args.n_env}')

        if (t+1) % int(args.save_freq) == 0:

            with open(f"./results/offline_{file_name}.pkl", 'wb') as f:
                pickle.dump(total_step_collection, f)
                print('results saved')

            if args.save_model:
                policy.save(f"./models/{file_name}")




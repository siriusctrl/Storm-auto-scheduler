import numpy as np
from numpy.lib.polynomial import polyint
import torch
import gym
import argparse
import os

import sys
sys.path.append(os.path.join(os.getcwd(), 'Simulator'))

import pickle

from WordCounting import WordCountingEnv
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
        state, done = eval_env.reset(), False
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
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=200, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)       # How often (time steps) we evaluate
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
    parser.add_argument("--max_episodic_length", default=50, type=int)              
    parser.add_argument("--n_env", default=10, type=int)
    parser.add_argument("--extra_name", default="")            
    args = parser.parse_args()

    file_name = f"parallel_{args.policy}_cSim_condense_{args.extra_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: cSim, Seed: {args.seed}, Parallel condense")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    parallel_env = ParalllelWrapper([WordCountingEnv(seed=args.seed) for _ in range(args.n_env)], args.n_env)
    eval_env = WordCountingEnv(seed=args.seed+100)
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
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # evaluations = [eval_policy(policy, args.env, args.seed)]
    evaluations = []
    states, done = parallel_env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_step_collection = {}

    for t in range(int(args.max_timesteps // args.n_env)):
        # episode_timesteps += t*args.n_env
        episode_timesteps += 1

        actions = []
        if t*args.n_env < args.start_timesteps:
            actions = [eval_env.action_space.sample() for _ in range(args.n_env)]
        else:
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
            done_bool = done if episode_timesteps < args.max_episodic_length else True
            # if done_bool == True:
            #     print(episode_timesteps, args.max_episodic_length)
            # Store in replay buffer
            # reward = -np.log(np.abs(reward))
            # the true is only for not done bool, which defines a infinit horizon problem
            replay_buffer.add(states[index], actions[index], next_state, reward, True)
            # print(info['pre_action'])
            # print(actions[index])
            episode_reward += reward
            batch_reward += reward
            for key in info.keys():
                if key in total_step_collection:
                    total_step_collection[key].append(info[key])
                else:
                    total_step_collection[key] = [info[key]]
            print(total_step_collection)

        print(f'batch:{t+1}/{int(args.max_timesteps // args.n_env)} avg reward is {batch_reward/args.n_env}')

        # Train agent after collecting sufficient data
        if (t*args.n_env) >= args.start_timesteps:
            # perform n time gradient update
            for _ in range(args.n_env):
                policy.train(replay_buffer, args.batch_size)
        
        if done_bool:
            print(f"Total T: {t*args.n_env+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            states, done = parallel_env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t+1) % int(args.eval_freq/args.n_env) == 0:
            evaluations.append(eval_policy(policy, eval_env))
            # np.save(f"./results/{file_name}", evaluations)

            with open(f"./results/{file_name}_step.pkl", 'wb') as f:
                pickle.dump(total_step_collection, f)
            with open(f"./results/{file_name}_eval.pkl", 'wb') as f:
                pickle.dump(evaluations, f)

            if args.save_model:
                policy.save(f"./models/{file_name}")




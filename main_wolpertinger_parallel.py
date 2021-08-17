import numpy as np
from numpy.lib.polynomial import polyint
import torch
import argparse
import os
import pickle

import sys
sys.path.append(os.path.join(os.getcwd(), 'Simulator'))

from WordCounting_Wolpertinger import WordCountingEnv
from ParallelWrapper import ParalllelWrapper

from SimpleBuffer import ReplayBuffer
from Li_Wolpertinger import Wolpertinger as li_wolp
from our_Wolpertinger import Wolpertinger as our_wolp

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, eval_episodes=1):
    avg_reward = 0.
    step_count = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action, _ = policy.select_action(np.array(state), noise=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            step_count += 1
            if step_count >= 10:
                break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} steps:{avg_reward/step_count:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='li')                    # this can be either li or our
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=200, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5000, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.4, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--max_episodic_length", default=50, type=int)              
    parser.add_argument("--n_env", default=5, type=int)
    parser.add_argument("--eval_only", default=False, type=bool)            
    args = parser.parse_args()

    file_name = f"{args.model}_Wolpertinger_cSim_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.model}, Env: cSim, Seed: {args.seed}")
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

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # TODO: fill these two None
        "n_machines": eval_env.n_machines,
        "n_exe":len(eval_env.topology.executor_flat),
        "discount": args.discount,
        "tau": args.tau,
    }
    
    # if args.policy == "DDPG":
    #     policy = DDPG(**kwargs)
    # elif args.policy == "TD3":
    #     kwargs["policy_noise"] = args.policy_noise * max_action
    #     kwargs["noise_clip"] = args.noise_clip * max_action
    #     kwargs["policy_freq"] = args.policy_freq
    #     policy = TD3(**kwargs)
    #     policy.min_action = min_action
    # else:
    #     raise NotImplementedError
    if args.model == 'li':
        policy = li_wolp(**kwargs)
    elif args.model == 'our':
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_freq
        policy = our_wolp(**kwargs)
    else:
        raise ValueError('Unknown model', args.model)
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    
    if args.eval_only:
        eval_policy(policy, eval_env)
        os.exit()

    if args.model == 'li':
        replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=5000)
    else:
        replay_buffer = ReplayBuffer(state_dim, action_dim)
    # evaluations = [eval_policy(policy, args.env, args.seed)]
    evaluations = []
    states, done = parallel_env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_step_collection = []

    for t in range(int(args.max_timesteps // args.n_env)):
        # episode_timesteps += t*args.n_env
        episode_timesteps += 1

        actions = []
        proto_actions = []
        if t*args.n_env < args.start_timesteps:
            for _ in range(args.n_env):
                ac, proto_ac = eval_env.random_action()
                actions.append(ac)
                proto_actions.append(proto_ac)
        else:
            for state in states:
                ac, proto_ac = policy.select_action(np.array(state))
                actions.append(ac)
                proto_actions.append(proto_ac)
        
        # next_states, reward, done, info = parallel_env.step_multiple(actions)
        res = parallel_env.step_multiple(actions)
        next_states = []
        batch_reward = 0
        # print(f"res={len(res)}, action={len(actions)}, proto={len(proto_actions)}")
        for index in range(len(res)):
            next_state, reward, done, info = res[index]
            next_states.append(next_state)
            done_bool = done if episode_timesteps < args.max_episodic_length else True
            # if done_bool == True:
            #     print(episode_timesteps, args.max_episodic_length)
            # Store in replay buffer
            if args.model == 'li':
                replay_buffer.add(states[index], actions[index], next_state, reward, done_bool)
            elif args.model == 'our':
                # replay_buffer.add(states[index], proto_actions[index], next_state, reward, done_bool)
                replay_buffer.add(states[index], actions[index], next_state, reward, done_bool)
            else:
                raise ValueError()
            # print(info['pre_action'])
            # print(actions[index])
            episode_reward += reward
            batch_reward += reward
            total_step_collection.append(reward)

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




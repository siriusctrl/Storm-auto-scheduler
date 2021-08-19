import numpy as np
from numpy.lib.polynomial import polyint
import torch
import argparse
import os
import pickle

import sys
sys.path.append(os.path.join(os.getcwd(), 'Simulator'))

from WordCounting_Wolpertinger import WordCountingEnv as wolp_word
from WordCounting import WordCountingEnv as con_word
from ParallelWrapper import ParalllelWrapper

from SimpleBuffer import ReplayBuffer

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
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds         
    parser.add_argument("--n_env", default=5, type=int)
    parser.add_argument("--max_timesteps", default=10000, type=int)
    parser.add_argument("--max_episodic_length", default=50, type=int) 
    parser.add_argument('--which', default='wolp')     
    args = parser.parse_args()

    file_name = f"random_{args.which}_{args.seed}"
    print("---------------------------------------")
    print(f"Random Transactions of {args.which} Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./offline"):
        os.makedirs("./offline")

    if args.which == 'wolp':
        parallel_env = ParalllelWrapper([wolp_word(seed=args.seed) for _ in range(args.n_env)], args.n_env)
        eval_env = wolp_word(seed=args.seed+100)
    elif args.which == 'con':
        parallel_env = ParalllelWrapper([con_word(seed=args.seed) for _ in range(args.n_env)], args.n_env)
        eval_env = con_word(seed=args.seed+100)
    else:
        raise ValueError(args.which)

    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=10000)
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
        for _ in range(args.n_env):
            if args.which == 'wolp':
                ac, proto_ac = eval_env.random_action()
                proto_actions.append(proto_ac)
            elif args.which == 'con':
                ac = eval_env.action_space.sample()
            else:
                raise ValueError()
            actions.append(ac)
        
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
            replay_buffer.add(states[index], actions[index], next_state, reward, True)
            # print(info['pre_action'])
            # print(actions[index])
            episode_reward += reward
            batch_reward += reward
            total_step_collection.append(reward)

        print(f'batch:{t+1}/{int(args.max_timesteps // args.n_env)} avg reward is {batch_reward/args.n_env}')
        
        if done_bool:
            print(f"Total T: {t*args.n_env+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            states, done = parallel_env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            with open(f"./offline/{file_name}.pkl", 'wb') as f:
                pickle.dump(replay_buffer, f)

    with open(f"./offline/{file_name}.pkl", 'wb') as f:
            pickle.dump(replay_buffer, f)


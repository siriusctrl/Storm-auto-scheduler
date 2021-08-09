import numpy as np
from numpy.lib.polynomial import polyint
import torch
import gym
import argparse
import os

import sys
sys.path.append(os.path.join(os.getcwd(), 'Simulator'))

from SimpleBuffer import ReplayBuffer
from WordCounting import WordCountingEnv
from ourDDPG import DDPG
from TD3 import TD3
from Wolptinger.ContinuousCartPole import ContinuousCartPoleEnv


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1):
    if env_name == 'c-carpole':
        eval_env = ContinuousCartPoleEnv()
    elif args.env == 'stream':
        eval_env = WordCountingEnv()
    else:
        eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)

    avg_reward = 0.
    step_count = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            step_count += 1
            if (env_name == 'stream') and (step_count >= 10):
                break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} steps:{avg_reward/step_count:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="stream")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=200, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3000, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--max_episodic_length", default=150, type=int)              
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if args.env == 'c-carpole':
        env = ContinuousCartPoleEnv()
    elif args.env == 'stream':
        env = WordCountingEnv()
    else:
        env = gym.make(args.env)
    
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

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
        policy.min_action = env.action_space.low[0]
    else:
        raise NotImplementedError
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
    
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # evaluations = [eval_policy(policy, args.env, args.seed)]
    evaluations = []
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
        
        next_state, reward, done, info = env.step(action)
        done_bool = done if episode_timesteps < args.max_episodic_length else True
        print('step reward is', reward)
        # Store in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward


        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        
        if done_bool:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t+1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            # np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")



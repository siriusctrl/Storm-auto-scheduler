from WordCounting import WordCountingEnv
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = WordCountingEnv()
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, buffer_size=10000, learning_starts=50)
print('start learning')
model.learn(total_timesteps=1000)
model.save("ddpg_scheduling")
env = model.get_env()

obs = env.reset()
i = 0
while i < 10:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    i += 1

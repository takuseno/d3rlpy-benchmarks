import gym
import time
import pybullet_envs
import numpy as np


env = gym.make('HopperBulletEnv-v0')

env.reset()

step_times = []
for _ in range(1000):
    start_time = time.time()
    env.step(env.action_space.sample())
    step_times.append(time.time() - start_time)

print('mean step time: ', np.mean(step_times))

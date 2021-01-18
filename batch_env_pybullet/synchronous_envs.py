import time
import argparse
import gym
import pybullet_envs
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=1)
    args = parser.parse_args()

    envs = []
    for _ in range(args.n_envs):
        envs.append(gym.make('HopperBulletEnv-v0'))

    observation_shape = envs[0].observation_space.shape
    action_size = envs[0].action_space.shape[0]

    for env in envs:
        env.reset()

    step_times = []
    for _ in range(1000):
        start_time = time.time()

        observations = np.empty((args.n_envs, *observation_shape))
        rewards = np.empty(args.n_envs)
        dones = np.empty(args.n_envs)
        infos = []

        actions = np.random.random((args.n_envs, action_size))

        for i, env in enumerate(envs):
            observation, reward, done, info = env.step(actions[i])
            observations[i] = observation
            rewards[i] = reward
            dones[i] = done
            infos.append(info)

        step_times.append(time.time() - start_time)

    print('meann step time: ', np.mean(step_times))

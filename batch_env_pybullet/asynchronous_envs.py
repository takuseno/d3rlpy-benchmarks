import time
import argparse
import gym
import pybullet_envs
import numpy as np

from d3rlpy.envs import BatchEnvWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=1)
    args = parser.parse_args()

    make_env = lambda: gym.make('HopperBulletEnv-v0')
    env = BatchEnvWrapper([make_env for _ in range(args.n_envs)])

    env.reset()

    step_times = []
    for _ in range(1000):
        start_time = time.time()

        actions = np.random.random((args.n_envs, env.action_space.shape[0]))
        env.step(actions)

        step_times.append(time.time() - start_time)

    print('meann step time: ', np.mean(step_times))

    env.close()

import time
import argparse
import gym
import d4rl_atari
import numpy as np

from d3rlpy.envs import BatchEnvWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=1)
    args = parser.parse_args()

    make_env = lambda: gym.make('breakout-mixed-v0')
    env = BatchEnvWrapper([make_env for _ in range(args.n_envs)])

    env.reset()

    step_times = []
    for _ in range(1000):
        start_time = time.time()

        actions = np.random.randint(env.action_space.n, size=args.n_envs)
        env.step(actions)

        step_times.append(time.time() - start_time)

    print('meann step time: ', np.mean(step_times))

    env.close()

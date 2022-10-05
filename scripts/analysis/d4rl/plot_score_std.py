import argparse

import matplotlib.pyplot as plt
import numpy as np

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--last_num", type=int)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = load_all_algos_d4rl_scores(args.env, args.dataset, exclude=["SAC", "TD3"])

    # compute standard deviation
    labels = []
    mean_stds = []
    std_stds = []
    for score in scores:
        std = np.std(score.scores[:, -args.last_num :], axis=1)
        mean_stds.append(np.mean(std))
        std_stds.append(np.std(std))
        labels.append(score.algo)

    x_values = np.arange(len(labels))

    plt.bar(x_values, mean_stds, yerr=std_stds, color="b", ecolor="black", width=0.3, align="center", capsize=5)
    plt.xticks(x_values, labels)
    plt.ylabel("average standard deviation")

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()

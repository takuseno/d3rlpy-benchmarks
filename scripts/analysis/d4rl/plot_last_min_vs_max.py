import argparse

import matplotlib.pyplot as plt
import numpy as np

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--last_num", type=int, default=1)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = load_all_algos_d4rl_scores(args.env, args.dataset, exclude=["CRR"])

    # compute diff of max and min scores
    diff_means = []
    diff_stds = []
    labels = []
    for score in scores:
        max_values = np.max(score.scores[:, -args.last_num :], axis=1)
        min_values = np.min(score.scores[:, -args.last_num :], axis=1)
        diff = max_values - min_values
        diff_means.append(np.mean(diff))
        diff_stds.append(np.std(diff))
        labels.append(score.algo)
    x_values = np.arange(len(labels))

    plt.bar(
        x_values,
        diff_means,
        yerr=diff_stds,
        color="b",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Mean Min-Max Gap in Last {args.last_num}",
    )
    plt.xticks(x_values, labels)
    plt.ylabel("Normalized Score")
    plt.ylim(bottom=0.0)
    plt.legend(loc="upper left")

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()

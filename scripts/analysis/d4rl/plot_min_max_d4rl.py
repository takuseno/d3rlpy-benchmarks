import argparse

import matplotlib.pyplot as plt
import numpy as np

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num", type=int)
    parser.add_argument("--last_num", type=int, default=1)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = load_all_algos_d4rl_scores(args.env, args.dataset, exclude=["CRR"])

    # compute topN and bottomN scores
    min_scores = []
    max_scores = []
    min_stddev = []
    max_stddev = []
    labels = []
    for score in scores:
        mean_scores = np.mean(score.scores[:, -args.last_num :], axis=1)
        min_score = np.sort(mean_scores)[: args.num]
        max_score = np.sort(mean_scores)[-args.num :]
        min_scores.append(np.mean(min_score))
        max_scores.append(np.mean(max_score))
        min_stddev.append(np.std(min_score))
        max_stddev.append(np.std(max_score))
        labels.append(score.algo)
    x_values = np.arange(len(labels))

    plt.bar(
        x_values,
        min_scores,
        yerr=min_stddev,
        color="b",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Bottom {args.num}",
    )
    plt.bar(
        x_values + 0.3,
        max_scores,
        yerr=max_stddev,
        color="r",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Top {args.num}",
    )
    plt.xticks(x_values + 0.3 / 2, labels)
    plt.ylabel("normalized score")
    plt.legend()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()

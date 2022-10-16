import argparse

import matplotlib.pyplot as plt

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores
from d3rlpy_benchmarks.plot_utils import plot_aggregate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--last_num", type=int, default=1)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = load_all_algos_d4rl_scores(args.env, args.dataset, exclude=["CRR"])
    plot_aggregate_metrics(scores, args.last_num)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()

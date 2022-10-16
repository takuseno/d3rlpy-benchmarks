import argparse

import matplotlib.pyplot as plt
import numpy as np

from d3rlpy_benchmarks.data_loader import ScoreData, load_all_algos_d4rl_scores
from d3rlpy_benchmarks.plot_utils import plot_aggregate_metrics

ENVS = ["hopper", "walker2d", "halfcheetah"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--last_num", type=int, default=1)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = []
    for env in ENVS:
        scores.extend(load_all_algos_d4rl_scores(env, args.dataset, exclude=["CRR"]))

    # gather same algorithms
    score_dict = {}
    for score in scores:
        if score.algo not in score_dict:
            score_dict[score.algo] = []
        score_dict[score.algo].append(score)

    # merge scores
    merged_scores = []
    for algo, scores in score_dict.items():
        merged_raw_score = np.array([score.raw_scores[:, -args.last_num :].reshape(-1) for score in scores])
        merged_score = np.array([score.scores[:, -args.last_num :].reshape(-1) for score in scores])
        merged_data = ScoreData(
            algo=algo,
            env="",
            steps=scores[0].steps,
            dataset=args.dataset,
            raw_scores=merged_raw_score,
            scores=merged_score,
        )
        merged_scores.append(merged_data)

    plot_aggregate_metrics(merged_scores, 10 * args.last_num)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()

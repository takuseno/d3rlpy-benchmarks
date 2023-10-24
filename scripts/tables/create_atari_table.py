import csv

import numpy as np

from d3rlpy_benchmarks.data_loader import get_atari_algo_list, get_atari_env_list, load_atari_score


def format_float(score):
    return "{:.1f}".format(score)


def main():
    with open("atari_table.csv", "w") as f:
        writer = csv.writer(f)

        header = [
            "algo",
            "env",
            "final return",
            "final std",
            "best return",
            "best std",
        ]
        writer.writerow(header)

        for algo in get_atari_algo_list():
            for env in get_atari_env_list():
                score = load_atari_score(algo, env)
                if len(score.scores) == 0:
                    continue
                final_returns = score.scores[:, -1]
                final_avg = np.mean(final_returns)
                final_std = np.std(final_returns)
                best_returns = np.max(score.scores, axis=1)
                best_avg = np.mean(best_returns)
                best_std = np.std(best_returns)
                row = [
                    algo,
                    env,
                    format_float(final_avg),
                    format_float(final_std),
                    format_float(best_avg),
                    format_float(best_std),
                ]
                print(row)
                writer.writerow(row)


if __name__ == "__main__":
    main()

import glob
import os
import csv

EXPERT_SCORES = {
    "halfcheetah": 12135.0,
    "hopper": 3234.3,
    "walker2d": 4592.3,
}

# calculated from d4rl paper
BASE_RANDOM = {
    "halfcheetah": -280.178953,
    "hopper": -20.272305,
    "walker2d": 1.629008,
}


def compute_normalized_score(raw_score, env):
    expert_score = EXPERT_SCORES[env]
    random_score = BASE_RANDOM[env]
    return (raw_score - random_score) / (expert_score - random_score)


def main():
    table = {}
    for log_dir in glob.glob("reproductions/*"):
        base = log_dir.split('/')[-1]
        splits = base.split('_')

        algo = splits[0]
        env = splits[1].split('-')[0]
        dataset = '-'.join(splits[1].split('-')[1:])

        if algo not in table:
            table[algo] = {}

        if env not in table[algo]:
            table[algo][env] = {}

        if dataset not in table[algo][env]:
            table[algo][env][dataset] = []

        with open(os.path.join(log_dir, 'environment.csv'), 'r') as f:
            reader = csv.reader(f)
            results = [row for row in reader]

        table[algo][env][dataset].append(float(results[-1][-1]))

    with open("table.csv", "w") as f:
        writer = csv.writer(f)

        header = ["algo", "env", "dataset", "return", "normalized return"]
        writer.writerow(header)

        for algo in table.keys():
            for env in table[algo].keys():
                for dataset in table[algo][env]:
                    returns = table[algo][env][dataset]
                    avg = sum(returns) / len(returns)
                    row = [algo, env, dataset, avg, 100.0 * compute_normalized_score(avg, env)]
                    print(row)
                    writer.writerow(row)


if __name__ == '__main__':
    main()

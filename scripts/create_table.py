import glob
import os
import csv


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

        header = ["algo", "env", "dataset", "return"]
        writer.writerow(header)

        for algo in table.keys():
            for env in table[algo].keys():
                for dataset in table[algo][env]:
                    returns = table[algo][env][dataset]
                    avg = sum(returns) / len(returns)
                    row = [algo, env, dataset, avg]
                    print(row)
                    writer.writerow(row)


if __name__ == '__main__':
    main()

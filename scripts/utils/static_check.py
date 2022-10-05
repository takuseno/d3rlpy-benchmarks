import subprocess


def execute(cmd):
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    execute(["black", "-l", "120", "d3rlpy_benchmarks", "scripts"])
    execute(["isort", "-l", "120", "--profile", "black", "d3rlpy_benchmarks", "scripts"])
    execute(["mypy", "d3rlpy_benchmarks"])


if __name__ == "__main__":
    main()

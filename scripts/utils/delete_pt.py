import glob
import os


def main():
    for path in glob.glob("reproductions/*/*.pt"):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob("extra/*/*/*.pt"):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob("d4rl/*/*/*.d3"):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob("atari/*/*.pt"):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob("finetuning/*/*.pt"):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob("baselines/plas/*/*.pth"):
        os.remove(path)
        print(f"{path} has been deleted.")
    for path in glob.glob("baselines/plas/*/*.npy"):
        os.remove(path)
        print(f"{path} has been deleted.")
    for path in glob.glob("baselines/plas/*/vae_logs.p"):
        os.remove(path)
        print(f"{path} has been deleted.")


if __name__ == "__main__":
    main()

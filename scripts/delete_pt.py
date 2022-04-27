import os
import glob


def main():
    for path in glob.glob('reproductions/*/*.pt'):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob('extra/*/*/*.pt'):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob('d4rl/*/*.pt'):
        os.remove(path)
        print(f"{path} has been deleted.")

    for path in glob.glob('atari/*/*.pt'):
        os.remove(path)
        print(f"{path} has been deleted.")


if __name__ == '__main__':
    main()

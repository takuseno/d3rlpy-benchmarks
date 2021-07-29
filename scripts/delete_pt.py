import os
import glob


def main():
    for path in glob.glob('reproductions/*/*.pt'):
        os.remove(path)
        print(f"{path} has been deleted.")


if __name__ == '__main__':
    main()
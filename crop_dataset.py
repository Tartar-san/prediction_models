import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=int, help="Server you use (1 - work, 2 - ucu)")
    args = parser.parse_args(["--server"])
import argparse
import json
import os

from glob import glob


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--name', type=str, required=True)

    config = args.parse_args()

    path = os.path.join('result', config.name)
    path = glob(os.path.join(path, f'result_*.json'))
    for i in sorted(path):
        with open(i, 'r') as f:
            results = json.load(f)

        print(i, len(results))


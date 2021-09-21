
import argparse
import json
import os


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True,
                  help='name of search')
args.add_argument('--num', type=int, required=True,
                  help='number')

metric = 'test_seld_score'

config = args.parse_args()

with open(f'result/{config.name}/result_{config.num}.json') as f:
    results = json.load(f)
results.sort(key=lambda x: x['perf']['objective_score'])

with open(f'model_config/{config.name}_{config.num}.json','w') as f:
    json.dump(results[0]['config'], f, indent=4)
    

import argparse
import json
import os


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True,
                  help='name must be {name}_{divided index} ex) 2021_1')

metric = 'test_seld_score'

config = args.parse_args()
if config.name.split('.')[-1] != 'json':
    config.name += '.json'

with open(config.name, 'r') as f:
    models = json.load(f)

del models['train_config']
bestmodel = sorted(models.items(), key=lambda x: x[1]['perf'][metric])
for i in range(1):
    with open(os.path.join('model_config', os.path.splitext(config.name)[0] + f'_best_model_{i+1}.json'), 'w') as f:
        json.dump(bestmodel[i][-1]['config'], f, indent=4)
    
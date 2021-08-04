import argparse
import json
import os
from glob import glob


class Writer:
    def __init__(self, train_config):
        self.abspath = '.'
        self.train_config = train_config
        self.result_path = self.make_dir(os.path.join('result', self.train_config.name))
        self.train_config_path = os.path.join(self.result_path, 'train_config.json')
        self.index = self.get_index()
        if self.train_config.new:
            os.system(f'rm -rf {os.path.join(self.result_path, "*")}')

    def train_config_dump(self):
        self.dump(self.train_config, self.train_config_path)

    def train_config_load(self):
        return self.load(self.train_config_path)

    def make_dir(self, path):
        if not os.path.exists(path):
            print(path)
            os.makedirs(path)
        return path

    def dump(self, content, path):
        if not isinstance(content, (list, tuple, dict, set)):
            try:
                content = vars(content)
            except:
                raise TypeError(f'content type is not a right type => {type(content)}')

        with open(path, 'w') as f:
            json.dump(content, f, indent=4)
    
    def load(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_index(self):
        previous_results = sorted(glob(os.path.join(self.result_path, f'result_*')))
        num_result = len(previous_results)
        if num_result != 0:
            index = num_result - 1
        else:
            index = 0
        return index
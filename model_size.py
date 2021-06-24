import os

import models
from params import get_param
from data_loader import load_seldnet_data, seldnet_data_to_dataloader
from transforms import split_total_labels_to_sed_doa
from trainv2 import get_dataset


def get_model_size(model):
    with open('model.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    parameters = 0
    with open('model.txt','r') as f:
        contents = f.read()
        i = 0
        j = 0
        while i < len(contents):
            if contents[i] == ' ':
                while True:
                    j += 1
                    if contents[i + j].isdigit():
                        continue
                    else:
                        if j == 1:
                            break
                        else:
                            if contents[i + j] == ' ':
                                parameters += int(contents[i + 1: i + j])
                                i += j - 1
                                break
                            else:
                                break
            i += 1
            j = 0
    os.system('rm -rf model.txt')
    return parameters


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    config, model_config = get_param()
    
    testset = get_dataset(config, 'test')
    x, y = [(x, y) for x, y in testset.take(1)][0]
    input_shape = x.shape
    model_config['n_classes'] = 12
    model = getattr(models, config.model)(input_shape, model_config)
    print(get_model_size(model))
    
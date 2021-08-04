def search_space_sanity_check(search_space: dict):
    '''
        search_space: {search_space_2d: ..., search_space_1d: ...}
    '''
    for name, val in search_space.items():
        if isinstance(val, dict):
            search_space_sanity_check(val)
        elif isinstance(val, (list, tuple)):
            if len(val) == 0:
                raise ValueError(f'len of value in {name} must be > 0')
        else:
            raise ValueError(f'values of {name} must be tuple or list')


def postprocess_fn(model_config):
    blocks = [i for i in model_config.keys() if isinstance(model_config.get(i, None), str) and 'mother_stage' in model_config.get(i, None)]

    for block in blocks:
        stage_type = model_config[block]

        if stage_type == 'mother_stage':
            args = model_config[f'{block}_ARGS']
            if args['filters2'] == 0:
                args['connect2'][2] = 1
                if args['filters0'] != 0:
                    args['connect2'][1] = 1

            if args['filters0'] == 0:
                args['connect0'][0] = 1
                args['kernel_size0'] = 0
                args['connect1'][1] = 0
                args['connect2'][1] = 0
                
            if args['filters2'] == 0:
                args['kernel_size2'] = 0

                if args['connect2'][2] == 0:
                    args['filters1'] = 0

                    if args['connect2'][1] == 0:
                        args['filters0'] = 0
    return model_config

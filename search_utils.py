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
    return model_config

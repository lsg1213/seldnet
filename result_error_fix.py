import json
from search import search_space_1d, search_space_2d


with open('result/loss/result_1.json','r') as f:
    results = json.load(f)

for result in results:
    blocks = [i for i in result['config'].keys() if i.startswith('BLOCK') and not 'ARGS' in i]
    keys = ['n_classes', 'BLOCK0', 'BLOCK0_ARGS', 'BLOCK1', 'BLOCK1_ARGS', 'BLOCK2', 'BLOCK2_ARGS', 'BLOCK3', 'BLOCK3_ARGS', 'SED', 'SED_ARGS', 'DOA', 'DOA_ARGS']

    if len(blocks) != 4:
        tmp = {}
        i = 0
        while i < len(keys):
            if keys[i].startswith('BLOCK'):
                for num in range(4):
                    v = result['config'].get(f'BLOCK{num}')
                    if v is None:
                        if num != 3:
                            tmp[keys[i]] = result['config'].get(f'BLOCK{num+1}')
                            tmp[keys[i + 1]] = result['config'].get(f'BLOCK{num+1}_ARGS')
                            del result['config'][f'BLOCK{num+1}']
                            del result['config'][f'BLOCK{num+1}_ARGS']
                            i += 2
                        else:
                            tmp[keys[i]] = 'identity_block'
                            tmp[keys[i + 1]] = {}
                            i += 2
                    else:
                        tmp[keys[i]] = v
                        tmp[keys[i + 1]] = result['config'].get(f'BLOCK{num}_ARGS')
                        i += 2
            else:
                tmp[keys[i]] = result['config'][keys[i]]
                i += 1
        result['config'] = tmp
    
    for block in ['BLOCK0', 'BLOCK1', 'BLOCK2', 'BLOCK3', 'SED', 'DOA']:
        args = result['config'].get(block)
        if args is None:
            raise ValueError('something wrong')
        search_space_1d.update(search_space_2d)
        if args == 'identity_block':
            continue
        sp = search_space_1d[args]
        for k in [i for i in result['config'][block + '_ARGS'].keys()]:
            v = result['config'][block + '_ARGS'].get(k)
            tmp = {}
            if not k in sp.keys():
                if args in ('simple_dense_stage', 'bidirectional_GRU_stage'):
                    newkey = args.split('_')[1] + '_' + k
                else:
                    newkey = '_'.join([args.split('_')[0],k])
                for key in result['config'][block + '_ARGS'].keys(): # 순서 맞추려고 이렇게 함
                    if k != key:
                        tmp[key] = result['config'][block + '_ARGS'][key]
                    else:
                        tmp[newkey] = result['config'][block + '_ARGS'][key]

                result['config'][block + '_ARGS'] = tmp
                
with open('result/loss/result_1.json','w') as f:
    json.dump(results, f, indent=4)

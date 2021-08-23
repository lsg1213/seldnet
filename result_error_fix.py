import json


with open('result/initial/result_3.json','r') as f:
    results = json.load(f)

for result in results:
    blocks = [i for i in result['config'].keys() if i.startswith('BLOCK') and not 'ARGS' in i]
    if len(blocks) != 4:
        tmp = {}
        keys = ['n_classes', 'BLOCK0', 'BLOCK0_ARGS', 'BLOCK1', 'BLOCK1_ARGS', 'BLOCK2', 'BLOCK2_ARGS', 'BLOCK3', 'BLOCK3_ARGS', 'SED', 'SED_ARGS', 'DOA', 'DOA_ARGS']
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
with open('result/initial/result_3.json','w') as f:
    json.dump(results, f, indent=4)
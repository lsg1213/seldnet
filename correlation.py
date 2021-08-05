import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import pandas as pd
import seaborn as sns
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--cor', action='store_true')
config = arg.parse_args()


def ema(data, n=10):
    emas = []
    for i in data:
        if len(emas) == 0:
            emas.append(i)
        else:
            emas.append(i * (2 / (1 + n)) + emas[-1] * (1 - (2 / (1 + n))))
    return emas

# 데이터
'''
    loss: [sample_num, epoch]
    seldscore: [sample_num,]
'''

def main():
    samples = sorted(glob('sampling_result/*.json'))
    val_dloss = []
    val_sloss = []
    val_emadloss = []
    val_emasloss = []
    test_seldscore = []

    for fp in samples:
        with open(fp, 'r') as f:
            result = json.load(f) # [model_config, val_slosses, val_dlosses, test_score]

        dema = ema(result[2])
        sema = ema(result[1])
        if config.cor:
            val_dloss.append([result[2][epoch - 1] for epoch in range(1,50)])
            val_sloss.append([result[1][epoch - 1] for epoch in range(1,50)])
            val_emadloss.append([dema[epoch - 1] - dema[epoch - 1 - 1] for epoch in range(1,50)])
            val_emasloss.append([sema[epoch - 1] - sema[epoch - 1 - 1] for epoch in range(1,50)])
            test_seldscore.append([np.array(result[3][-51]) for epoch in range(1,50)])
        else:
            epoch = 10
            val_dloss.append(result[2][epoch - 1])
            val_sloss.append(result[1][epoch - 1])
            val_emadloss.append(dema[epoch - 1] - dema[epoch - 1 - 1])
            val_emasloss.append(sema[epoch - 1] - sema[epoch - 1 - 1])
            test_seldscore.append(result[3][-51])
    val_sloss = np.array(val_sloss) # (model num, epoch,)
    val_dloss = np.array(val_dloss)
    val_emadloss = np.array(val_emadloss)
    val_emasloss = np.array(val_emasloss)
    test_seldscore = np.array(test_seldscore) # (model num,)

    total_loss = 1000 * np.array(val_dloss) + np.array(val_sloss)
     # x : (model num, valloss) y: (model num, score)
    if config.cor:   
        data = np.corrcoef(np.concatenate([val_emasloss + val_emadloss * 1000, test_seldscore[...,:1]], -1).T)
        # data = np.corrcoef(np.concatenate([total_loss, test_seldscore[...,:1]], -1).T)
        row_indices = [str(i) for i in range(1,50)] + ['test_score']
        
        column_names = [str(i) for i in range(1,50)] + ['test_score']
        data_df = pd.DataFrame(data, index=row_indices, columns=column_names)
        corr = data_df.corr()
        sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
        plt.show()
        

        return
    test_seldscore = np.stack(test_seldscore)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.scatter(test_seldscore, np.minimum(total_loss, 45), norm=False)
    ax2.scatter(test_seldscore, np.minimum(val_dloss, val_dloss.min() * 2), norm=False)
    ax3.scatter(test_seldscore, np.minimum(val_sloss, val_sloss.min() * 2), norm=False)
    ax4.scatter(test_seldscore, val_emasloss + val_emadloss * 1000)
    ax5.scatter(test_seldscore, val_emadloss)
    ax6.scatter(test_seldscore, val_emasloss)

    ax1.set_ylabel('weighted total_loss')
    ax2.set_ylabel('DOA loss')
    ax3.set_ylabel('SED loss')
    ax4.set_ylabel('weighted total EMA')
    ax5.set_ylabel('DOA ema')
    ax6.set_ylabel('SED ema')
    ax1.set_xlabel('final seld score, low is good')
    ax2.set_xlabel('final seld score, low is good')
    ax3.set_xlabel('final seld score, low is good')
    ax4.set_xlabel('final seld score, low is good')
    ax5.set_xlabel('final seld score, low is good')
    ax6.set_xlabel('final seld score, low is good')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
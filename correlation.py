import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
import pandas as pd
import seaborn as sns
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--cor', action='store_true')
arg.add_argument('--outlier', action='store_true')
arg.add_argument('--epoch', type=int, default=10)
arg.add_argument('--start', type=int, default=1)
arg.add_argument('--end', type=int, default=50)
config = arg.parse_args()


def ema(data, n=4):
    emas = []
    for i in data:
        if len(emas) == 0:
            emas.append(i)
        else:
            emas.append(i * (2 / (1 + n)) + emas[-1] * (1 - (2 / (1 + n))))
    return emas

def get_objective_score(val_finalscore, total_loss, val_ema_subtract_loss):
    # score = weights[0] * (val_seld_score) +\
    #         weights[1] * exp(K[-1] - K[-2]) +\
    #         weights[2] * exp(val_loss[-1] - val_loss[0])
    val_subtract_loss = np.copy(total_loss)

    for i in range(0, total_loss.shape[-1]):
        if i == 0:
            val_subtract_loss[:,i] = 1
        else:
            val_subtract_loss[:,i] = total_loss[:,i] - total_loss[:,0]
    return val_finalscore

'''
    loss: [sample_num, epoch]
    seldscore: [sample_num,]
'''

def main():
    start_epoch, end_epoch = config.start, config.end
    samples = sorted(glob('sampling_result/*.json'))
    val_dloss = []
    val_sloss = []
    val_emadloss = []
    val_emasloss = []
    val_score = []
    val_finalscore = []
    val_ema_subtract_loss = []
    test_seldscore = []

    for fp in samples:
        with open(fp, 'r') as f:
            result = json.load(f) # [model_config, val_slosses, val_dlosses, test_score, val_score]
        if len(result) == 4:
            os.system(f'rm -rf {fp}')
        # if result[3][-51] > 0.95:
        #     continue
        
        dema = ema(result[2])
        sema = ema(result[1])
        totalema = ema(result[2] * 1000 + result[1])
        val_dloss.append([result[2][epoch - 1] for epoch in range(start_epoch, end_epoch)])
        val_sloss.append([result[1][epoch - 1] for epoch in range(start_epoch, end_epoch)])
        val_ema_subtract_loss.append([totalema[epoch - 1] - totalema[epoch - 1 - 1] if epoch > 1 else 0 for epoch in range(start_epoch, end_epoch)])
        val_emadloss.append([dema[epoch - 1] - dema[epoch - 1 - 1] if epoch > 1 else 0 for epoch in range(start_epoch, end_epoch)])
        val_emasloss.append([sema[epoch - 1] - sema[epoch - 1 - 1] if epoch > 1 else 0 for epoch in range(start_epoch, end_epoch)])
        test_seldscore.append([np.array(result[3][-51]) for epoch in range(start_epoch, end_epoch)])
        val_score.append([np.array(result[4][epoch - 1 - 1]) for epoch in range(start_epoch, end_epoch)])
        val_finalscore.append([np.array(result[4][-51]) for epoch in range(start_epoch, end_epoch)])

    val_sloss = np.array(val_sloss) # (model num, epoch,)
    val_dloss = np.array(val_dloss)
    val_emadloss = np.array(val_emadloss)
    val_emasloss = np.array(val_emasloss)
    val_ema_subtract_loss = np.array(val_ema_subtract_loss)
    val_score = np.array(val_score)
    test_seldscore = np.array(test_seldscore) # (model num,)
    val_finalscore = np.array(val_finalscore)

    total_loss = 1000 * np.array(val_dloss) + np.array(val_sloss)
    
    # x : (model num, valloss) y: (model num, score)
    objective_score = get_objective_score(val_finalscore, total_loss, val_ema_subtract_loss)

    # filter
    if config.outlier:
        condition = np.where(objective_score[:,7] < 3)
        objective_score = objective_score[condition]
        test_seldscore = test_seldscore[condition]
        val_sloss = val_sloss[condition]
        val_dloss = val_dloss[condition]
        val_emadloss = val_emadloss[condition]
        val_emasloss = val_emasloss[condition]
        val_ema_subtract_loss = val_ema_subtract_loss[condition]
        val_score = val_score[condition]
        val_finalscore = val_finalscore[condition]
        total_loss = total_loss[condition]


    if config.cor:   
        # data = np.corrcoef(np.concatenate([val_emasloss + val_emadloss * 1000, test_seldscore[...,:1]], -1).T)
        data = np.corrcoef(np.concatenate([objective_score[...,1:], test_seldscore[...,:1]], -1).T)
        row_indices = [str(i) for i in range(start_epoch + 1, end_epoch)] + ['test_score']
        column_names = [str(i) for i in range(start_epoch + 1, end_epoch)] + ['test_score']
        data_df = pd.DataFrame(data, index=row_indices, columns=column_names)
        corr = data_df.corr()
        sns.heatmap(corr,
            # annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
        plt.show()
        return
    val_sloss = val_sloss[:, config.epoch - 1]
    val_dloss = val_dloss[:, config.epoch - 1]
    val_emadloss = val_emadloss[:, config.epoch - 1]
    val_emasloss = val_emasloss[:, config.epoch - 1]
    val_ema_subtract_loss = val_ema_subtract_loss[:, config.epoch - 1]
    val_score = val_score[:, config.epoch - 1]
    test_seldscore = test_seldscore[:, config.epoch - 1]
    objective_score = objective_score[:, config.epoch - 1]
    total_loss = total_loss[:, config.epoch - 1]
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.scatter(test_seldscore, np.minimum(total_loss, 45), norm=False)
    ax2.scatter(test_seldscore, objective_score, norm=False)
    ax3.scatter(test_seldscore, np.minimum(val_sloss, val_sloss.min() * 2), norm=False)
    ax4.scatter(test_seldscore, val_emasloss + val_emadloss * 1000)
    ax5.scatter(test_seldscore, val_emadloss)
    ax6.scatter(test_seldscore, val_score)

    ax1.set_ylabel('weighted total_loss')
    ax2.set_ylabel('objective score')
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
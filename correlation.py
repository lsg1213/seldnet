import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
import pandas as pd
import seaborn as sns
import argparse
from scipy.stats import spearmanr
from numpy import exp, log10, sign, absolute


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

def get_objective_score(finalscore, total_loss, ema_subtract_loss):
    finalscore = finalscore[:,:1]
    weights = [1, 1, 1]

    decreased_loss = [total_loss[:, step] - total_loss[:, 0] for step in range(total_loss.shape[1])]
    decreased_loss = np.stack(decreased_loss, -1)
    # decreased_loss[np.where(decreased_loss == 0)] = 1e-4

    finalscore = weights[0] * finalscore
    # finalscore /= np.linalg.norm(finalscore)
    ema_subtracted_loss = weights[1] * ema_subtract_loss
    # ema_subtracted_loss /= np.linalg.norm(ema_subtracted_loss)
    decreased_loss = weights[2] * decreased_loss
    decreased_loss = sign(decreased_loss) * log10(absolute(decreased_loss))
    # decreased_loss /= np.linalg.norm(decreased_loss)
    score = finalscore 
    return score

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
    train_finalscore = []
    val_finalscore = []
    val_ema_subtract_loss = []
    train_ema_subtract_loss = []
    test_seldscore = []
    train_dloss = []
    train_sloss = []
    train_score = []


    for fp in samples:
        with open(fp, 'r') as f:
            # model_config, train_slosses, train_dlosses, val_slosses, val_dlosses, train_scores, test_score, val_score
            result = json.load(f) # [model_config, val_slosses, val_dlosses, test_score, val_score]
        if len(result) == 4:
            os.system(f'rm -rf {fp}')
        # if result[3][-51] > 0.95:
        #     continue
        val_dema = ema(result[4])
        val_sema = ema(result[3])
        train_dema = ema(result[2])
        train_sema = ema(result[1])
        val_totalema = ema(result[4] * 1000 + result[3])
        val_dloss.append([result[4][epoch - 1] for epoch in range(start_epoch - 1, end_epoch)])
        val_sloss.append([result[3][epoch - 1] for epoch in range(start_epoch - 1, end_epoch)])
        val_ema_subtract_loss.append([val_totalema[epoch - 1] - val_totalema[epoch - 1 - 1] if epoch > 1 else 0 for epoch in range(start_epoch - 1, end_epoch)])
        train_ema_subtract_loss.append([val_totalema[epoch - 1] - val_totalema[epoch - 1 - 1] if epoch > 1 else 0 for epoch in range(start_epoch - 1, end_epoch)])
        val_emadloss.append([val_dema[epoch] - val_dema[epoch - 1] if epoch > 1 else 0 for epoch in range(start_epoch - 1, end_epoch)])
        val_emasloss.append([val_sema[epoch] - val_sema[epoch - 1] if epoch > 1 else 0 for epoch in range(start_epoch - 1, end_epoch)])
        test_seldscore.append([np.array(result[6][-1]) for epoch in range(start_epoch - 1, end_epoch)])
        val_score.append([np.array(result[7][epoch - 1]) for epoch in range(start_epoch - 1, end_epoch)])
        val_finalscore.append([np.array(result[7][-1]) for epoch in range(start_epoch - 1, end_epoch)])
        train_dloss.append([result[2][epoch - 1] for epoch in range(start_epoch - 1, end_epoch)])
        train_sloss.append([result[1][epoch - 1] for epoch in range(start_epoch - 1, end_epoch)])
        train_score.append([np.array(result[5][epoch - 1]) for epoch in range(start_epoch - 1, end_epoch)])
        train_finalscore.append([np.array(result[5][-1]) for epoch in range(start_epoch - 1, end_epoch)])


    val_sloss = np.array(val_sloss) # (model num, epoch,)
    val_dloss = np.array(val_dloss)
    val_emadloss = np.array(val_emadloss)
    val_emasloss = np.array(val_emasloss)
    train_ema_subtract_loss = np.array(train_ema_subtract_loss)
    val_ema_subtract_loss = np.array(val_ema_subtract_loss)
    val_score = np.array(val_score)
    test_seldscore = np.array(test_seldscore) # (model num,)
    train_seldscore = np.array(train_score)
    train_finalscore = np.array(train_finalscore)
    val_finalscore = np.array(val_finalscore)
    train_dloss = np.array(train_dloss)
    train_sloss = np.array(train_sloss)
    train_score = np.array(train_score)

    total_val_loss = 1000 * np.array(val_dloss) + np.array(val_sloss)
    total_train_loss = 1000 * train_dloss + train_sloss
    
    # x : (model num, valloss) y: (model num, score)
    val_objective_score = get_objective_score(val_finalscore, total_val_loss, val_ema_subtract_loss)
    train_objective_score = get_objective_score(train_finalscore, total_train_loss, train_ema_subtract_loss)
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
        total_val_loss = total_val_loss[condition]
        total_train_loss = total_train_loss[condition]
        train_dloss = train_dloss[condition]
        train_sloss = train_sloss[condition]
        train_score = train_score[condition]


    if config.cor:   
        # data = np.corrcoef(np.concatenate([val_emasloss + val_emadloss * 1000, test_seldscore[...,:1]], -1).T)
        # data = np.corrcoef(np.concatenate([total_val_loss[...,1:], test_seldscore[...,:1]], -1).T)
        # data, p_value = spearmanr(np.concatenate([objective_score[...,1:], test_seldscore[...,:1]], -1))
        data, p_value = spearmanr(np.concatenate([train_objective_score[..., 1:], test_seldscore[...,:1]], -1))
        data2, p_value2 = spearmanr(np.concatenate([val_objective_score[..., 1:], test_seldscore[...,:1]], -1))
        row_indices = [str(i) for i in range(start_epoch + 1, end_epoch + 1)] + ['test_score']
        column_names = ['test_score']
        # data_df = pd.DataFrame(data[...,-1], index=row_indices, columns=column_names)
        # corr = data_df
        # sns.heatmap(corr,
        #     # annot=True,
        #     xticklabels=corr.columns.values,
        #     yticklabels=corr.columns.values,vmin=-1, vmax=1)
        data = data[:-1, -1]
        data2 = data2[:-1, -1]
        line1, = plt.plot(list(range(2, len(data) + 2)), data, color='b', label='train_score')
        line2, = plt.plot(list(range(2, len(data2) + 2)), data2, color='r', label='val_score')
        line3, = plt.plot(list(range(2, len(data) + 2)), np.ones(len(data)) * 0.7, color='g')
        plt.xticks(range(2, len(data) + 2))
        plt.yticks([i / 100 for i in list(range(40, 100, 5))])
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('correlation')
        print('p_value:', p_value[-1])
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
    total_val_loss = total_val_loss[:, config.epoch - 1]
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.scatter(test_seldscore, np.minimum(total_val_loss, 45), norm=False)
    ax2.scatter(test_seldscore, objective_score, norm=False)
    ax3.scatter(test_seldscore, np.minimum(val_sloss, val_sloss.min() * 2), norm=False)
    ax4.scatter(test_seldscore, val_emasloss + val_emadloss * 1000)
    ax5.scatter(test_seldscore, val_emadloss)
    ax6.scatter(test_seldscore, val_score)

    ax1.set_ylabel('weighted total_val_loss')
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
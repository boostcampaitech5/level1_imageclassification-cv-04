import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, num_classes, normalize=False, save_path=None):
    if normalize:
        n_total = torch.sum(cm, 1).view(num_classes, 1)
        np_cm = cm / n_total
        np_cm = np_cm.numpy()
        ax = sns.heatmap(np_cm, annot=True, cmap='Blues', linewidth=.5,
                        fmt=".2f", annot_kws = {'size' : 6})
    else:
        np_cm = cm.numpy()
        ax = sns.heatmap(np_cm, annot=True, cmap='Blues', linewidth=.5,
                        fmt="d", annot_kws = {'size' : 6})

    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels([i for i in range(num_classes)])
    ax.xaxis.tick_top()
    ax.yaxis.set_ticklabels([i for i in range(num_classes)])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return ax


def cnt_per_classes(df_path):
    info_df = pd.read_csv('./input/data/train/train_info.csv')
    class_counts = info_df["ans"].value_counts()
    class_counts.sort_index(inplace= True)

    return class_counts.tolist()

if __name__ == '__main__':
    
    print(cnt_per_classes("./input/data/train/train_info.csv"))
    # num_classes = 3
    # cm = [[5, 1, 1],
    #       [0, 5, 2],
    #       [1, 0, 5]]
    # cm = torch.FloatTensor(cm)
    # fig = plot_confusion_matrix(cm, 3, normalize=True)
    # plt.savefig('./test_imgs/plot_cm_normalize.png')
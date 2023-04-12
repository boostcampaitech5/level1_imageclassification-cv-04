import torch
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, num_classes, normalize=False):
    if normalize:
        n_total = torch.sum(cm).item()
        cm = cm / n_total
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels([f'Class_{i}' for i in range(num_classes)])
    ax.yaxis.set_ticklabels([f'Class_{i}' for i in range(num_classes)])
    return ax


if __name__ == '__main__':
    num_classes = 3
    cm = [[5, 1, 1],
          [0, 5, 2],
          [1, 0, 5]]
    cm = torch.FloatTensor(cm)
    fig = plot_confusion_matrix(cm, 3, normalize=True)
    plt.savefig('./test_imgs/plot_cm_normalize.png')
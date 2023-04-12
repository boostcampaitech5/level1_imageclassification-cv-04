import torch

def confusion_matrix(model, data_iter, device, num_classes):
    with torch.no_grad():
        model.eval()
        cm = torch.zeros((num_classes, num_classes))
        for data, target in data_iter:
            model_pred, y_target = model(data.to(device)), target.to(device)
            _, y_pred = torch.max(model_pred.data, 1)
            if y_pred == y_target:
                cm[y_target][y_target] += 1
            else:
                cm[y_target][y_pred] += 1

    return cm


def accuracy(cm, num_classes):
    n_total, n_correct = torch.sum(cm), 0
    for i in range(num_classes):
        n_correct += cm[i][i]
    acc = (n_correct/n_total)

    return acc.item()


def precision(cm, num_classes):
    precision_val = torch.sum(cm, 0) # TP + FP
    for i in range(num_classes):
        precision_val[i] = cm[i][i] / precision_val[i] # TP / (TP + FP)
    precision_val = torch.sum(precision_val).item() / num_classes

    return precision_val


def recall(cm, num_classes):
    recall_val = torch.sum(cm, 1) # TP + FN
    for i in range(num_classes):
        recall_val[i] = cm[i][i] / recall_val[i] # TP / (TP + FN)
    recall_val = torch.sum(recall_val).item() / num_classes

    return recall_val


def f1_score(cm, num_classes):
    precision_val = precision(cm, num_classes)
    recall_val = recall(cm, num_classes)
    f1_score_val = 2 * ((precision_val * recall_val) / (precision_val + recall_val))

    return f1_score_val
    

if __name__ == '__main__':
    num_classes = 3
    cm = [[5, 1, 1],
          [0, 5, 2],
          [1, 0, 5]]
    cm = torch.FloatTensor(cm)
    print(accuracy(cm, num_classes))
    print(precision(cm, num_classes))
    print(recall(cm, num_classes))
    print(f1_score(cm, num_classes))




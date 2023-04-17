import torch

def confusion_matrix(model, data_iter, device, num_classes,convert=False):
    with torch.no_grad():
        model.eval()
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for data, target in data_iter:
            model_pred, y_target = model(data.to(device)), target.to(device)
            #_, y_pred = torch.max(model_pred.data, 1)
            if convert:
                model_pred = convert_pred(model_pred)
                #print(model_pred[0],y_target[0])
                model_pred = convert_class(model_pred)
                y_target = convert_class(y_target)
           # print(model_pred,y_target)
            for y_p, y_t in zip(model_pred, y_target):
                if y_p.data == y_t.data:
                    cm[y_t][y_t] += 1
                else:
                    cm[y_t][y_p] += 1

    return cm
def convert_pred(pred,convert=False):
    class_pred = pred[:,:2].argmax(dim=1)
    mask_pred = pred[:5:].argmax(dim=1)
    age_pred = pred[:,2:5].argmax(dim=1)


    return torch.cat((class_pred.view(-1,1),age_pred.view(-1,1),mask_pred.view(-1,1)),dim=1).long()

def convert_class(logit):
    # bucket = torch.Tensor([30,60]).cuda()
    # age_pred = torch.bucketize(logit[:,1],bucket,right=True)
    classes = torch.zeros(len(logit),dtype=torch.long).cuda()
    classes += 6*logit[:,2]
    classes += 3*logit[:,0]
    classes += logit[:,1]
    return classes
    

def accuracy(cm, num_classes):
    n_total, n_correct = torch.sum(cm), 0
    for i in range(num_classes):
        n_correct += cm[i][i]
    acc = (n_correct/n_total)

    return acc.item()


def precision(cm, num_classes):
    predict_positive = torch.sum(cm, 0) # TP + FP
    precision_val = torch.zeros(num_classes)
    for i in range(num_classes):
        val = cm[i][i] / predict_positive[i] # TP / (TP + FP)
        if not torch.isnan(val):
            precision_val[i] = val
    precision_val = torch.sum(precision_val).item() / num_classes

    return precision_val


def recall(cm, num_classes):
    actual_positive = torch.sum(cm, 1) # TP + FN
    recall_val = torch.zeros(num_classes)
    for i in range(num_classes):
        val = cm[i][i] / actual_positive[i] # TP / (TP + FN)
        if not torch.isnan(val):
            recall_val[i] = val
    recall_val = torch.sum(recall_val).item() / num_classes

    return recall_val


def f1_score(cm, num_classes):
    precision_val = precision(cm, num_classes)
    recall_val = recall(cm, num_classes)
    if precision_val + recall_val == 0:
        return 0
    f1_score_val = 2 * ((precision_val * recall_val) / (precision_val + recall_val))

    return f1_score_val
    

if __name__ == '__main__':
    # num_classes = 3

    # cm = [[5, 1, 1],
    #       [0, 5, 2],
    #       [1, 0, 5]]
    # cm = torch.LongTensor(cm)
    # # print(accuracy(cm, num_classes))
    # # print(precision(cm, num_classes))
    # # print(recall(cm, num_classes))
    # # print(f1_score(cm, num_classes))
    logit = torch.Tensor([[-3,3,63.5,-1,-0.8,3]]).cuda()
    label = torch.Tensor([[1,61,2]]).long().cuda()
    l=convert_pred(logit)
    print(l)
    l=convert_class(l)
    print(l)
    l=convert_class(label)
    print(l)



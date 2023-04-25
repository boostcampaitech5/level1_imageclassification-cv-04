import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import numpy as np
from sklearn.metrics import confusion_matrix
_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class cmMetter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.pred = None
        self.label = None
    def update(self,pred,label):
        if type(self.pred) != np.ndarray:
            self.pred = pred.cpu().detach().numpy().reshape(-1)
            self.label = label
        else:
            self.pred = np.concatenate((self.pred,pred.cpu().detach().numpy().reshape(-1)))
            self.label = np.concatenate((self.label, label.cpu().detach().numpy()))

def toConfusionMatrix(y_pred, y_label, num_classes:int) -> np.ndarray:
    
    cm = confusion_matrix(y_label,y_pred, labels = np.arange(num_classes).tolist())
    #cm[y_pred][y_gt]
    return cm
def predToClass(pred):
    #pred가 단일 클래스 추측이 아닐경우
    return pred

def train(model, dataloader, criterion, optimizer, device: str, args) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    cm_m = cmMetter()
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)

        loss = criterion(outputs, targets)    
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        cm_m.update(preds, targets)
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
        
        batch_time_m.update(time.time() - end)
    
        if idx % args.log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
        confusionmatrix = toConfusionMatrix(cm_m.pred, cm_m.label,args.num_classes)
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg), ('cm',confusionmatrix)])
        
def test(model, dataloader, criterion, device: str, args) -> dict:
    correct = 0
    total = 0
    total_loss = 0
    cm_m = cmMetter()
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            cm_m.update(preds,targets)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            if idx % args.log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
        confusionmatrix = toConfusionMatrix(cm_m.pred,cm_m.label)
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader)),('cm',confusionmatrix)])
                
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, 
    savedir: str, device: str, args
) -> None:

    best_acc = 0
    step = 0
    for epoch in range(args.epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{args.epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, args.log_interval, device,args)
        eval_metrics = test(model, testloader, criterion, args.log_interval, device)

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        if args.use_wandb:
            wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

            #save confusion_matrix
            if args.use_cm:
                fig = plot_confusion_matrix(eval_metrics['cm'],args.num_classes)
                if args.user_wandb:
                    wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")},step=epoch)

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))

#임시로 여기 작성
import matplotlib as plt
import seaborn as sns
def plot_confusion_matrix(cm, num_classes, normalize=False, save_path=None):

    plt.clf()
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



if __name__ == '__main__':
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    cm_m = cmMetter()
    end = time.time()
    class testClass():
        def __init__(self):
            super().__init__()
        def __len__(self):
            return 3
        def __getitem__(self,idx):
            pred = torch.Tensor([[1],[2],[3]])
            label=torch.Tensor([1,2,4])
            return pred,label
    testset = testClass()
    for idx, (inputs, targets) in enumerate(testset):

        data_time_m.update(time.time() - end)
        # predict
        preds = targets
        # accuracy
        cm_m.update(inputs, targets)
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
        batch_time_m.update(time.time() - end) 
        end = time.time()
        if idx==2:
            break
    print(cm_m.pred)
    print(cm_m.label)
    confusion_matrix = toConfusionMatrix(cm_m.pred, cm_m.label,5)
    print(confusion_matrix)



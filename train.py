import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import numpy as np
from utils.train_util import plot_confusion_matrix,toConfusionMatrix
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
    #1epoch까지의 결과 저장
    def __init__(self):
        self.reset()
    def reset(self):
        self.pred = None
        self.label = None
    def update(self,pred,label):
        if self.pred.shape(-1) != 0:
            raise TypeError('Model\'s pred shape is not match with data label')
        if type(self.pred) != np.ndarray:
            self.pred = pred.cpu().detach().numpy().reshape(-1)
            self.label = label
        else:
            self.pred = np.concatenate((self.pred,pred.cpu().detach().numpy().reshape(-1)))
            self.label = np.concatenate((self.label, label.cpu().detach().numpy()))


def outputToPred(outputs):
    #output -> 단일 클래스 pred로 변환
    return outputs.argmax(dim=1)

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
        preds = outputToPred(outputs)
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
            preds = outputToPred(outputs)
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
        eval_metrics = test(model, testloader, criterion, args.log_interval, device,args)

        # wandb
        # cm은 매번 저장되지 않도록 metric -> metric[:-1]로 수정
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics[:-1].items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics[:-1].items()])
        if args.use_wandb:
            wandb.log(metrics, step=epoch)

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
                if args.use_wandb:
                    wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")},step=epoch)
                
    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))




#test후 삭제
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



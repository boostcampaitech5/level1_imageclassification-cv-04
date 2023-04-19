from dataloader import *
from model import *
from metric import *
from utils import *
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchsummary import summary
import multiprocessing
from tqdm import tqdm


def run(args):
    csv_path = os.path.join(args.eval_path, 'info.csv')
    save_csv_path = 'eval_info.csv'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The device is ready\t>>\t{device}')

    # Image size 조절과 tensor로만 만들어주면 됨(normalize까지는 해야 할 듯)
    transform = transforms.Compose([transforms.CenterCrop((384, 384)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])

    dataset = ClassificationDataset(csv_path = csv_path,
                                    transform=transform,
                                    train=False,
                                    eval_path=args.eval_path)
    print(f'The number of testing images\t>>\t{len(dataset)}')

    test_iter = DataLoader(dataset,
                           batch_size=args.batch_size,
                           num_workers=multiprocessing.cpu_count() // 2)
    
    print('Loading checkpoint ...')
    state_dict0 = torch.load(args.checkpoint0)
    state_dict1 = torch.load(args.checkpoint1)
    state_dict2 = torch.load(args.checkpoint2)
    state_dict3 = torch.load(args.checkpoint3)
    state_dict4 = torch.load(args.checkpoint4)

    print('The model is ready ...')
    model0 = Classifier(args).to(device)
    model0.load_state_dict(state_dict0['model_state_dict'])
    model1 = Classifier(args).to(device)
    model1.load_state_dict(state_dict1['model_state_dict'])
    model2 = Classifier(args).to(device)
    model2.load_state_dict(state_dict2['model_state_dict'])
    model3 = Classifier(args).to(device)
    model3.load_state_dict(state_dict3['model_state_dict'])
    model4 = Classifier(args).to(device)
    model4.load_state_dict(state_dict4['model_state_dict'])


    print("Starting testing ...")
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    result = []
    for test_img, _ in tqdm(test_iter):
        with torch.no_grad():
            test_img = test_img.to(device)
            class_pred = torch.zeros((args.batch_size, args.num_classes)).to(device)
            for model in [model0, model1, model2, model3, model4]:
                test_pred = model(test_img)
                class_pred.add_(test_pred)
            _, max_pred = torch.max(class_pred, 1)
            result.append(max_pred.item())


    print('Save CSV file')
    df = pd.read_csv(csv_path)
    df['ans'] = result
    df.to_csv(save_csv_path, index=False)


if __name__ == '__main__':
    args_dict = {'eval_path' : '../input/data/eval',
                 'checkpoint0' : './checkpoint/kfold4_0_focal_reducelr27_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.816)_loss(0.318)_f1(0.763)_state_dict.pt',
                 'checkpoint1' : './checkpoint/kfold4_1_focal_reducelr28_bs64_ep100_adamw_lr0.0001_resnet50/epoch(45)_acc(0.773)_loss(0.325)_f1(0.687)_state_dict.pt',
                 'checkpoint2' : './checkpoint/kfold4_2_focal_reducelr29_bs64_ep100_adamw_lr0.0001_resnet50/epoch(29)_acc(0.740)_loss(0.354)_f1(0.648)_state_dict.pt',
                 'checkpoint3' : './checkpoint/kfold4_3_focal_reducelr30_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.758)_loss(0.290)_f1(0.683)_state_dict.pt',
                 'checkpoint4' : './checkpoint/kfold4_4_focal_reducelr31_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.770)_loss(0.409)_f1(0.670)_state_dict.pt',
                 'load_model':'resnet50',
                 'load_mode' : 'state_dict',
                 'num_classes' : 18,
                 'batch_size' : 1,
                 'model_summary' : False}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args)
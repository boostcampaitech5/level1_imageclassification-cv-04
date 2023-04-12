from dataloader import *
from model import *
from metric import *
from utils import *
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchsummary import summary


def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The device is ready\t>>\t{device}')

    # Image size 조절과 tensor로만 만들어주면 됨(normalize까지는 해야 할 듯)
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = args.csv_path,
                                    transform=transform,
                                    train=False)
    print(f'The number of testing images\t>>\t{len(dataset)}')

    test_iter = DataLoader(dataset,
                           batch_size=args.batch_size)
    
    if args.load_mode == 'state_dict':
        print('Loading checkpoint ...')
        state_dict = torch.load(args.checkpoint)

        print('The model is ready ...')
        model = Network(num_classes = args.num_classes).to(device)
        if args.model_summary:
            print(summary(model, (3, 256, 256)))
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        print('The model is ready ...')
        model = torch.load(args.checkpoint).to(device)
        if args.model_summary:
            print(summary(model, (3, 256, 256)))

    print("Starting testing ...")
    model.eval()
    result = []
    for test_img, test_target in test_iter:
        test_img, test_target = test_img.to(device), test_target.to(device)
        test_pred = model(test_img)
        _, max_pred = torch.max(test_pred, 1)
        result.append(max_pred.item())

    print('Save CSV file')
    df = pd.read_csv(args.csv_path)
    df['ans'] = result
    df.to_csv(args.save_csv_path, index=False)


if __name__ == '__main__':
    args_dict = {'csv_path' : './input/data/eval/info.csv',
                 'save_csv_path' : './input/data/eval/eval_info.csv',
                 'checkpoint' : './checkpoint/epoch(0)_acc(0.366)_loss(3.851)_f1(0.182)_model.pt',
                 'load_mode' : 'model',
                 'pt_path' : './checkpoint',
                 'num_classes' : 18,
                 'batch_size' : 1,
                 'model_summary' : True}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args)
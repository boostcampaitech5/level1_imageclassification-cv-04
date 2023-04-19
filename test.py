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
    transform = transforms.Compose([transforms.CenterCrop(384),
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
    state_dict = torch.load(args.checkpoint)

    print('The model is ready ...')
    model = Classifier(args).to(device)
    if args.model_summary:
        print(summary(model, (3, 384, 384)))
    model.load_state_dict(state_dict['model_state_dict'])

    print("Starting testing ...")
    model.eval()
    result = []
    for test_img, _ in tqdm(test_iter):
        with torch.no_grad():
            test_img = test_img.to(device)
            test_pred = model(test_img)
            _, max_pred = torch.max(test_pred, 1)
            result.append(max_pred.item())

    print('Save CSV file')
    df = pd.read_csv(csv_path)
    df['ans'] = result
    df.to_csv(save_csv_path, index=False)


if __name__ == '__main__':
    args_dict = {'eval_path' : '../input/data/eval',
                 'checkpoint' : './checkpoint/kfold4_0_focal_reducelr27_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.816)_loss(0.318)_f1(0.763)_state_dict.pt',
                 'load_model':'resnet50',
                 'load_mode' : 'state_dict',
                 'num_classes' : 18,
                 'batch_size' : 1,
                 'model_summary' : True}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args)
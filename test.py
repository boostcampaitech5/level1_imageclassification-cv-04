from dataloader import *
from model import *
from metric import *
from utils import *
import wandb_info
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchsummary import summary
import multiprocessing
from tqdm import tqdm

def run(args):
    csv_path = os.path.join(args.eval_path, 'info.csv')
    save_csv_path = os.path.join(args.eval_path, f'eval_info_{args.wandb_exp_name}_{args.exp_num}.csv')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'The device_age is ready\t>>\t{device}')

    # Image size 조절과 tensor로만 만들어주면 됨(normalize까지는 해야 할 듯)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    #transforms.CenterCrop(384, 384),#(256, 256),
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
    
    print('The model is ready ...')
    model_mask = Classifier2(args.load_model, args.num_mask_classes).to(device)
    model_gender = Classifier2(args.load_model, args.num_gender_classes).to(device)
    model_age = Classifier2(args.load_model, args.num_age_classes).to(device)

    if args.load_mode == 'state_dict':
        print('Loading checkpoint ...')
        state_dict = torch.load(args.checkpoint)        
        model_mask.load_state_dict(state_dict['model_mask_state_dict'])
        model_gender.load_state_dict(state_dict['model_gender_state_dict'])
        model_age.load_state_dict(state_dict['model_age_state_dict'])
    else:
        # to load model pickle. (not implemented)
        pass

    print("Starting testing ...")
    model_mask.eval()
    model_gender.eval()
    model_age.eval()
    result = []

    pbar_test = tqdm(test_iter)
    for _, (test_img, _) in enumerate(pbar_test):
        pbar_test.set_description(f"Test. iter:")
        with torch.no_grad():
            test_img = test_img.to(device)
            test_mask_pred = model_mask(test_img)
            test_gender_pred = model_gender(test_img)
            test_age_pred = model_age(test_img)
            test_pred = torch.max(test_mask_pred, 1)[1] * 6 + torch.max(test_gender_pred, 1)[1] * 3 + torch.max(test_age_pred, 1)[1]
            result.append(test_pred.item())
    pbar_test.close()

    df = pd.read_csv(csv_path)
    df['ans'] = result
    df.to_csv(save_csv_path, index=False)
    print('Save CSV file', save_csv_path)

if __name__ == '__main__':
    args_dict = {'eval_path' : '../input/data/eval',
                 'checkpoint' : './checkpoint/seplearn_ViT_resize_224_no_crop_sepplots_kfold_65_bs32_ep100_adamw_lr5e-06_vit_small_patch16_224/best_state_dict copy.pt', 
                 'load_mode' : 'state_dict', #'model'
                 'load_model' : 'vit_small_patch16_224', #'resnet18',
                 'num_classes' : 18,
                 'num_mask_classes' : 3,
                 'num_gender_classes' : 2,
                 'num_age_classes' : 3,
                 'batch_size' : 1,
                 'model_summary' : True}
    
    wandb_data = wandb_info.get_wandb_info()
    args_dict.update(wandb_data)
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args)
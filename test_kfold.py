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
    transform = transforms.Compose([transforms.CenterCrop((300, 300)),
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
    

    output = [[], [], []]
    for i, (checkpoint, num_classes) in enumerate(zip([args.mask_checkpoint, args.gender_checkpoint, args.age_checkpoint], [args.num_mask, args.num_gender, args.num_age])):
        print('Loading checkpoint ...')
        state_dict0 = torch.load(checkpoint[0])
        state_dict1 = torch.load(checkpoint[1])
        state_dict2 = torch.load(checkpoint[2])
        state_dict3 = torch.load(checkpoint[3])
        state_dict4 = torch.load(checkpoint[4])

        print('The model is ready ...')
        model0 = KFoldClassifier(num_classes, args.load_model[i]).to(device)
        model0.load_state_dict(state_dict0['model_state_dict'])
        model1 = KFoldClassifier(num_classes, args.load_model[i]).to(device)
        model1.load_state_dict(state_dict1['model_state_dict'])
        model2 = KFoldClassifier(num_classes, args.load_model[i]).to(device)
        model2.load_state_dict(state_dict2['model_state_dict'])
        model3 = KFoldClassifier(num_classes, args.load_model[i]).to(device)
        model3.load_state_dict(state_dict3['model_state_dict'])
        model4 = KFoldClassifier(num_classes, args.load_model[i]).to(device)
        model4.load_state_dict(state_dict4['model_state_dict'])


        print("Starting testing ...")
        model0.eval()
        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        for test_img, _ in tqdm(test_iter):
            with torch.no_grad():
                test_img = test_img.to(device)
                class_pred = torch.zeros((args.batch_size, num_classes)).to(device)
                for model in [model0, model1, model2, model3, model4]:
                    test_pred = model(test_img)
                    class_pred.add_(test_pred)
                _, max_pred = torch.max(class_pred, 1)
                output[i].append(max_pred.item())

    result = []
    for i in range(len(output[0])):
        mask, gender, age = output[0][i], output[1][i], output[2][i]
        result.append(mask*6 + gender*3 + age)


    print('Save CSV file')
    df = pd.read_csv(csv_path)
    df['ans'] = result
    df.to_csv(save_csv_path, index=False)


if __name__ == '__main__':
    args_dict = {'eval_path' : '../input/data/eval',
                 'mask_checkpoint' : ['/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_maskdetection_reducelr61_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.984)_loss(0.057)_f1(0.984)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_1_cd_maskdetection_reducelr67_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.991)_loss(0.039)_f1(0.991)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_2_cd_maskdetection_reducelr69_bs64_ep100_adamw_lr0.0001_resnet50/epoch(54)_acc(0.991)_loss(0.051)_f1(0.991)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_3_cd_maskdetection_reducelr73_bs64_ep100_adamw_lr0.0001_resnet50/epoch(49)_acc(0.991)_loss(0.032)_f1(0.991)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_4_cd_maskdetection_reducelr76_bs64_ep100_adamw_lr0.0001_resnet50/epoch(29)_acc(0.992)_loss(0.043)_f1(0.992)_state_dict.pt'],
                 'gender_checkpoint' : ['/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_genderdetection_reducelr66_bs64_ep100_adamw_lr0.0001_resnet50/epoch(56)_acc(0.944)_loss(0.158)_f1(0.941)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_1_cd_genderdetection_reducelr70_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.965)_loss(0.114)_f1(0.963)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_2_cd_genderdetection_reducelr70_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.967)_loss(0.111)_f1(0.966)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_3_cd_genderdetection_reducelr74_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.953)_loss(0.137)_f1(0.950)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_4_cd_genderdetection_reducelr77_bs64_ep100_adamw_lr0.0001_resnet50/epoch(42)_acc(0.923)_loss(0.216)_f1(0.919)_state_dict.pt'],
                 'age_checkpoint' : ['/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_agedetection_reducelr65_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.884)_loss(0.316)_f1(0.853)_state_dict.pt',
                                     '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_agedetection_reducelr65_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.884)_loss(0.316)_f1(0.853)_state_dict.pt',
                                     '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_agedetection_reducelr65_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.884)_loss(0.316)_f1(0.853)_state_dict.pt',
                                     '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_agedetection_reducelr65_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.884)_loss(0.316)_f1(0.853)_state_dict.pt',
                                     '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_agedetection_reducelr65_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.884)_loss(0.316)_f1(0.853)_state_dict.pt'],
                 'load_model':['resnet50', 'resnet50', 'resnet50'],
                 'load_mode' : 'state_dict',
                 'num_mask' : 3,
                 'num_gender' : 2,
                 'num_age' : 3,
                 'batch_size' : 1,
                 'model_summary' : False}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args)
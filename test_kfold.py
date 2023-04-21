from dataloader import *
from model import *
from metric import *
from utils import *
from torch.utils.data import DataLoader
from torchvision import transforms 
from tqdm import tqdm


def run(args):
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(args.test_dir, 'info.csv'))
    image_dir = os.path.join(args.test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([transforms.CenterCrop((300, 300)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size)

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_ckps = [args.mask_ckp, args.gender_ckp, args.age_ckp]
    num_classes = [args.num_mask, args.num_gender, args.num_age]
    device = torch.device('cuda')
    predictions = [[], [], []]

    for i, (ckps, num_class) in enumerate(zip(all_ckps, num_classes)):
        models = []
        for ckp in ckps:
            state_dict = torch.load(ckp)
            model = KFoldClassifier(num_class, args.load_model[i]).to(device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            models.append(model)

        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(device)
                # class_pred = torch.zeros((batch_size, num_class)).to(device)
                class_pred = []
                for model in models:
                    pred = model(images)
                    class_pred.append(pred)
                all_class_pred = torch.stack(class_pred)
                class_pred = torch.sum(all_class_pred, dim=0)
                pred = class_pred.argmax(dim=-1)
                predictions[i].extend(pred.cpu().numpy())

    predictions = np.array(predictions)
    all_predictions = predictions[0]*6 + predictions[1]*3 + predictions[2]
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(args.test_dir, 'submission.csv'), index=False)
    print('test inference is done!')


if __name__ == '__main__':
    args_dict = {'test_dir' : '/opt/ml/input/data/eval',
                 'mask_ckp' : ['/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_maskdetection_reducelr61_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.984)_loss(0.057)_f1(0.984)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_1_cd_maskdetection_reducelr67_bs64_ep100_adamw_lr0.0001_resnet50/epoch(39)_acc(0.991)_loss(0.039)_f1(0.991)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_2_cd_maskdetection_reducelr69_bs64_ep100_adamw_lr0.0001_resnet50/epoch(54)_acc(0.991)_loss(0.051)_f1(0.991)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_3_cd_maskdetection_reducelr73_bs64_ep100_adamw_lr0.0001_resnet50/epoch(49)_acc(0.991)_loss(0.032)_f1(0.991)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_4_cd_maskdetection_reducelr76_bs64_ep100_adamw_lr0.0001_resnet50/epoch(29)_acc(0.992)_loss(0.043)_f1(0.992)_state_dict.pt'],
                 'gender_ckp' : ['/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_genderdetection_reducelr66_bs64_ep100_adamw_lr0.0001_resnet50/epoch(56)_acc(0.944)_loss(0.158)_f1(0.941)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_1_cd_genderdetection_reducelr70_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.965)_loss(0.114)_f1(0.963)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_2_cd_genderdetection_reducelr70_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.967)_loss(0.111)_f1(0.966)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_3_cd_genderdetection_reducelr74_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.953)_loss(0.137)_f1(0.950)_state_dict.pt',
                                      '/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_4_cd_genderdetection_reducelr77_bs64_ep100_adamw_lr0.0001_resnet50/epoch(42)_acc(0.923)_loss(0.216)_f1(0.919)_state_dict.pt'],
                 'age_ckp' : ['/opt/ml/level1_imageclassification-cv-04/checkpoint/kfold4_0_cd_agedetection_reducelr65_bs64_ep100_adamw_lr0.0001_resnet50/epoch(19)_acc(0.884)_loss(0.316)_f1(0.853)_state_dict.pt',
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
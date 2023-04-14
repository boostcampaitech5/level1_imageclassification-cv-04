# level1_imageclassification-cv-04
level1_imageclassification-cv-04 created by GitHub Classroom

## Data Preprocessing
### train_info.csv
- Columns : ['ImageID', 'ans']
- ImageID
    - train_info.csv는 train 폴더 안에 넣어주시면 됩니다.(train.csv와 함께)
    - 현재 작업 폴더 : ~ ml
        - ./input/data/train/images/000001_female_Asian_...
- ans
    - label

## Root
```bash
├── input
│   ├── data
│   ├── eval
│   │   ├── images
│   │   ├── info.csv
│   │   └── eval_info.csv
│   └── train
│       ├── images
│       ├── train.csv
│       └── train_info.csv
└── code
    ├── dataloader
    │   ├── __init__.py
    │   └── dataset.py
    ├── metric
    │   ├── __init__.py
    │   └── metric.py
    ├── model
    │   ├── __init__.py
    │   └── model.py
    ├── utils
    │   ├── __init__.py
    │   ├── plot.py
    │   ├── sampler.py
    │   └── transform.py
    ├── transform_list.json
    ├── train.py
    └── test.py
```
## Accelerator 설치방법
---
``` bash
pip install accelerate
```
-  (필수) 기본 세팅 -> 
    https://github.com/TooTouch/Pytorch-Accelerator-Test 
- Accelerator 를 사용하게 되면 device 를 항상 추척(?)하기 때문에 to(device) 를 따로 안해줘도 됨

## TODO

- Custom Model
- Optimizer Setting
- Further More
    - GradCAM

## Argparse
### Train
|Argument|Description|Default|Possible value|
|---|---|---|---|
|seed|---|223|---|
|csv_path|train_info.csv 위치|./input/data/train/train_info.csv|---|
|save_path|parameter 저장 위치|./checkpoint|---|
|use_wandb|---|False|True, False|
|wandb_exp_name|---|test|---|
|wandb_project_name|---|Image_classification_mask|---|
|wandb_entity|---|connect-cv-04|---|
|num_classes|Class 개수|18|---|
|model_summary|모델 summary를 출력할지|False|True, False|
|batch_size|---|128|---|
|learning_rate|---|1e-4|---|
|epochs|---|1|---|
|train_val_split|train_set의 비율(val_set의 비율은 자동)|0.8|0 ~ 1|
|save_mode|state_dict를 저장할 지 model 자체를 저장할 지|both|state_dict, model, both|
|save_epoch|parameter를 몇 epoch마다 저장할지(Confusion matrix도 적용)|1|---|
|load_model|어떤 backbone network를 사용할 것인 지|resnet50|resnet50|
|transform_path|transform_list.json 파일의 위치치|./transform_list.json|---|
|transform_list|적용할 transform의 종류와 순서|['resize', 'totensor', 'normalize']|transform_list.json 참고|
|not_freeze_layer|freeze 할 layer|['layer4']|모델을 전부 fine tuning 하기로 해서 더 이상 사용 안함|
|weight_decay|optimizer에 들어갈 weight_decay|1e-2|---|

### Test
|Argument|Description|Default|Possible value|
|---|---|---|---|
|eval_path|eval folder 위치|./input/data/eval|---|
|checkpoint|저장된 parameter 위치|./checkpoint/epoch(0)_acc(0.366)_loss(3.851)_f1(0.182)_model.pt|---|
|load_mode|저장된 parameter를 불러오는 방식|model|state_dict, model|
|num_classes|Class 개수|18|---|
|batch_size|Test이므로 batch_size 1로 고정|1|---|
|model_summary|모델 summary를 출력할지|False|True, False|

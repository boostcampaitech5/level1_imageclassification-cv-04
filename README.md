# level1_imageclassification-cv-04
level1_imageclassification-cv-04 created by GitHub Classroom

## Root
```bash
├── input
│   └── data
│       ├── eval
│       │   ├── images
│       │   ├── info.csv
│       │   └── eval_info.csv
│       └── train
│           ├── images
│           ├── train.csv
│           └── train_info.csv
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
│   └── plot.py
├── train.py
└── test.py
```

## TODO

- Custom Model
- Optimizer Setting

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
|train_val_split|train_set의 비율(val_set의 비율은 자동)|0.8|---|
|save_mode|state_dict를 저장할 지 model 자체를 저장할 지|both|state_dict, both|
|save_epoch|parameter를 몇 epoch마다 저장할지(Confusion matrix도 적용)|1|---|

### Test
|Argument|Description|Default|Possible value|
|---|---|---|---|
|eval_path|eval folder 위치|./input/data/eval|---|
|checkpoint|저장된 parameter 위치|./checkpoint/epoch(0)_acc(0.366)_loss(3.851)_f1(0.182)_model.pt|---|
|load_mode|저장된 parameter를 불러오는 방식|model|state_dict, model|
|num_classes|Class 개수|18|---|
|batch_size|---|1|---|
|model_summary|모델 summary를 출력할지|False|True, False|
# level1_imageclassification-cv-04
level1_imageclassification-cv-04 created by GitHub Classroom

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

## run experiment
- `chmod 755 run.sh & ./run.sh`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`
- `SM_CHANNEL_TRAIN='/opt/ml/input/data/train/images' SM_MODEL_DIR='/opt/ml/code/model' python train.py`
- `SM_CHANNEL_TRAIN='/opt/ml/input/data/train/images' SM_MODEL_DIR='/opt/ml/CV-04_DH/model' python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`
- `SM_CHANNEL_EVAL='/opt/ml/input/data/eval' SM_CHANNEL_MODEL='/opt/ml/code/model/exp' SM_OUTPUT_DATA_DIR='/opt/ml/code/output' python inference.py`
- `SM_CHANNEL_EVAL='/opt/ml/input/data/eval' SM_CHANNEL_MODEL='/opt/ml/CV-04_DH/model/exp' SM_OUTPUT_DATA_DIR='/opt/ml/CV-04_daehee/output' python inference.py`
- `SM_CHANNEL_EVAL='/opt/ml/input/data/eval' SM_CHANNEL_MODEL='/opt/ml/CV-04_DH/model/exp' SM_OUTPUT_DATA_DIR='/opt/ml/output' python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`
- `SM_GROUND_TRUTH_DIR='/opt/ml/gt' SM_OUTPUT_DATA_DIR='/opt/ml/code/output' python evaluation.py`
- `SM_GROUND_TRUTH_DIR='/opt/ml/gt' SM_OUTPUT_DATA_DIR='/opt/ml/CV-04_DH/output' python evaluation.py`
- `SM_GROUND_TRUTH_DIR='/opt/ml/gt' SM_OUTPUT_DATA_DIR='/opt/ml/output' python evaluation.py`
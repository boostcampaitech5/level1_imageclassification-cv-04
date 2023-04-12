# level1_imageclassification-cv-04
level1_imageclassification-cv-04 created by GitHub Classroom


##동적 모듈 가져오기
모든 모듈의 이름은 snake case를 사용합니다. ex. base_model
클래스의 이름은 pascal case를 사용합니다. ex. BaseModel

사용자는 train set과 val set을 나누는 코드를 작성해야 합니다.\
해당 부분은 dataloader의 custom dataloader에서 make_dataset을 통해 구현해야 합니다.\
##config 작성 양식\
예시\

'''json
{
    "model":{
        "name": "mask_classification",
        "args": {"loss": "Custom"}
    },
    "dataset":{
        "split": false,
        "split_ratio": 0.8,

        "name": "mask_dataset",
        "args": {
            "base_dir":"data"
        },
        "annotation_dir": "/annotation"
    },
    "dataloader":{
        "batch_size": 1,
        "shuffle": false
    },
    "epochs": 10,
    "optimizer":{
        "name": "Adam",
        "args":{
            "lr": 0.01
        }
    }
}
'''

YOLO와 같이 logit에서 여러 loss를 적용시켜야 할 때가 있기 때문에, 이 부분을 사용자가 커스텀 할 수 있도록 model부분에서 구현할 수 있습니다.
custom loss를 사용할 경우 BaseModel의 custom_loss를 오버라이딩하면 됩니다.
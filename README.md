# level1_imageclassification-cv-04
level1_imageclassification-cv-04 created by GitHub Classroom


##동적 모듈 가져오기
모든 모듈의 이름은 snake case를 사용합니다. ex. base_model
클래스의 이름은 pascal case를 사용합니다. ex. BaseModel


##config 작성 양식
학습용 데이터는 data config에 작성
만약 valid set이 따로 존재하지 않을 경우
 "dataset":{
        "already_split": false,
        "split_ratio": 0.8,

        "name": "sample_dataset",
        "args": {
            "base_dir":"data"
        },
        "batch_size": 16,
        "annotation_dir": "/annotaion.csv"
    }
존재할 경우
"dataset":{
        "already_split": true,
        "train_dataset":{
            "name": "sample_dataset",
            "args": {
                "base_dir":"data"
            },
            "batch_size": 16,
            "annotation_dir": "/annotaion.csv"
        },
        "valid_dataset":{
            "name": "sample_dataset",
            "args": {
                "base_dir":"data/valid"
            },
            "batch_size": 16,
            "annotation_dir": "/annotaion.csv"
        }
    }
와 같이 작성할 것
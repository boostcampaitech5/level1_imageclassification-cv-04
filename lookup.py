import timm
from torchsummary import summary
from dataloader import *
from utils import *

for model in timm.list_models():
    if 'vit' in model:
        print(model)

# dataset = ClassificationDataset(csv_path = '../input/data/train/train_info.csv', transform=None)
# train_set, val_set, train_idx, val_idx = train_valid_split_by_sklearn(dataset, 0.8, 223)
# train_cnt = dataset.df['ans'][train_idx].value_counts().sort_index()
# normedWeights = [1 - (x / sum(train_cnt)) for x in train_cnt]

# df_mask = dataset.df['ans'][train_idx] // 6
# df_gender = (dataset.df['ans'][train_idx] // 3) % 2
# df_age = dataset.df['ans'][train_idx] % 3

# train_mask_cnt = df_mask.value_counts().sort_index()
# train_gender_cnt = df_gender.value_counts().sort_index()
# train_age_cnt = df_age.value_counts().sort_index()

# print(dataset.df['ans'][train_idx])
# print(train_cnt)
# print(sum(train_cnt))
# for x in train_cnt:
#     print(x)
# print(normedWeights)

# print(train_mask_cnt)
# print(train_gender_cnt)
# print(train_age_cnt)

# normedWeights_mask = [1 - (x / sum(train_mask_cnt)) for x in train_mask_cnt]
# normedWeights_gender = [1 - (x / sum(train_gender_cnt)) for x in train_gender_cnt]
# normedWeights_age = [1 - (x / sum(train_age_cnt)) for x in train_age_cnt]

# print(normedWeights_mask)
# print(normedWeights_gender)
# print(normedWeights_age)

# print(15120/18900)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model2 = timm.create_model('vit_base_patch8_224_in21k', pretrained=True, num_classes=3).to(device)
# print(summary(model2, (3, 224, 224)))
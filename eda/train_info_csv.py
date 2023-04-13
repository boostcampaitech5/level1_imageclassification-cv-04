import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


info_df = pd.read_csv('./input/data/train/train_info.csv')

fig, axes = plt.subplots(2, 3, figsize=(12,8))

img_paths = {key:'' for key in range(18)}
for label_id in range(3):
    for row in range(2):
        for col in range(3):
            random_id = np.random.randint(0,83, 1)
            img_path = info_df[info_df['ans'] == 6*label_id+2*row+col].iloc[random_id].ImageID
            img_paths[6*label_id+3*row+col] = img_path.iloc[0]
            axes[row][col].imshow(np.array(Image.open(img_path.iloc[0])))
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            axes[row][col].set_title(f'Class: {6*label_id+3*row+col}')
    fig.tight_layout()
    fig.savefig(f'./EDA_test_img/train_info_csv/img_ex{label_id}.png')

with open('./EDA_test_img/train_info_csv/img_ex.txt', 'w') as f:
    for key, val in img_paths.items():
        f.write(str(key))
        f.write(val)




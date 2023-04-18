import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import pickle
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import json


def get_img_stats(img_dir, img_ids):
    """
    데이터셋에 있는 이미지들의 크기와 RGB 평균 및 표준편차를 수집하는 함수
    
    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로
        img_ids: 학습 데이터셋 하위폴더 이름들

    Returns:
        img_info: 이미지들의 정보 (크기, 평균, 표준편차)
    """
    img_info = dict(heights=[], widths=[], means=[], stds=[])
    for img_id in tqdm(img_ids):
        for path in glob(os.path.join(img_dir, img_id, '*')):
            print(path)
            img = np.array(Image.open(path))
            h, w, _ = img.shape
            img_info['heights'].append(h)
            img_info['widths'].append(w)
            img_info['means'].append(img.mean(axis=(0,1)))
            img_info['stds'].append(img.std(axis=(0,1)))

    return img_info


def get_ext(img_dir, img_id):
    """
    학습 데이터셋 이미지 폴더에는 여러 하위폴더로 구성되고, 
    이 하위폴더들에는 각 사람의 사진들이 들어가있습니다. 
    하위폴더에 속한 이미지의 확장자를 구하는 함수입니다.
    
    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로 
        img_id: 학습 데이터셋 하위폴더 이름

    Returns:
        exts: 이미지의 확장자
    """
    exts = set()
    for filename in os.listdir(os.path.join(img_dir, img_id)):
        exts.add(os.path.splitext(filename)[-1].lower())

    return list(exts)


def count_face_detection(img_dir, img_ids):
    """
    OpenCV를 사용해 Face Detection 수행 했을 때 
    detection 되지 않는 얼굴들의 번호를 수집

    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로 
        img_ids: 학습 데이터셋 하위폴더 이름
    Returns:
        not_found: Face Detection 되지 않는 이미지 수
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    num2class = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
                 'mask4', 'mask5', 'normal']
    not_found = {k: [] for k in num2class}

    for img_id in tqdm(img_ids):
        exts = get_ext(img_dir, img_id)
        for i, class_id in enumerate(num2class):
            for ext in exts:
                try:
                    img_path = os.path.join(cfg.img_dir, img_id, class_id+ext)
                    img = np.array(Image.open(img_path))
                    break
                except:
                    continue
            bbox = face_cascade.detectMultiScale(img)
            if len(bbox) == 0:
                not_found[class_id].append(img_path)

    return not_found

def plot_mask_images(img_dir, idx, save_path):
    img_id = df.iloc[idx].path
    gen = df.iloc[idx].gender
    age = df.iloc[idx].age

    exts = get_ext(img_dir, img_id)
    imgs = []
    for class_name in num2class[:-1]:
        for ext in exts:
            try:
                img = np.array(Image.open(os.path.join(img_dir, img_id, class_name + ext))) 
                break
            except:
                continue
        imgs.append(img)
    
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15, 12))
    for i in range(n_rows*n_cols):
        axes[i//(n_rows+1)][i%n_cols].imshow(imgs[i])
        axes[i//(n_rows+1)][i%n_cols].set_title(f'{num2class[i]}-{gen}-{age}', color='r')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'mask{idx}.jpg'))


if __name__ == '__main__':

    save_path = './EDA_test_img'
    stats = False
    object_detection = False
    target_df = False
    xy = False
    not_detection_img = False
    n_not_detection = False
    no_mask_not_detection = False
    no_mask_noise = False
    mask_noise = False
    make_csv = False
    test_make_csv = False
    tSNE = False
    class_distribution = False
    make_split_csv = False
    make_limit_csv = False
    transforms_test = True
    save_csv_path = '../input/data/train/train_info.csv'

    class cfg:
        data_dir = '../input/data/train'
        img_dir = f'{data_dir}/images'
        df_path = f'{data_dir}/train.csv'

    num2class = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
                 'mask4', 'mask5', 'normal']
    class2num = {k: v for v, k in enumerate(num2class)}

    df = pd.read_csv(cfg.df_path)

    print(df.head())
    
    if stats:
        img_info = get_img_stats(cfg.img_dir, df.path.values)

        print(f'Total number of people is {len(df)}')
        print(f'Total number of images is {len(df) * 7}')

        print(f'Minimum height for dataset is {np.min(img_info["heights"])}')
        print(f'Maximum height for dataset is {np.max(img_info["heights"])}')
        print(f'Average height for dataset is {int(np.mean(img_info["heights"]))}')
        print(f'Minimum width for dataset is {np.min(img_info["widths"])}')
        print(f'Maximum width for dataset is {np.max(img_info["widths"])}')
        print(f'Average width for dataset is {int(np.mean(img_info["widths"]))}')

        print(f'RGB Mean: {np.mean(img_info["means"], axis=0) / 255.}')
        print(f'RGB Standard Deviation: {np.mean(img_info["stds"], axis=0) / 255.}')

    if object_detection:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        imgs = []
        img_id = df.iloc[500].path
        ext = get_ext(cfg.img_dir, img_id)
        for class_id in num2class:
            img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, class_id+ext)))
            imgs.append(img)
        imgs = np.array(imgs)

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 6))
        for ax_id, img_id in enumerate([0, 1, -1]):
            # img_id(0, 1, -1) -> ((턱스크, 코스크), 마스크, 노마스크)
            axes[ax_id].imshow(imgs[img_id])
            axes[ax_id].set_xticks([])
            axes[ax_id].set_yticks([])
        fig.savefig(os.path.join(save_path, 'HarrCascade_img1.png'))

    if target_df:
        # Gender
        fig = plt.figure(figsize=(6, 4.5)) 
        ax = sns.countplot(x = 'gender', data = df, palette=["#55967e", "#263959"])

        plt.xticks( np.arange(2), ['female', 'male'] )
        plt.title('Sex Ratio',fontsize= 14)
        plt.xlabel('')
        plt.ylabel('Number of images')

        counts = df['gender'].value_counts()
        counts_pct = [f'{elem * 100:.2f}%' for elem in counts / counts.sum()]
        for i, v in enumerate(counts_pct):
            ax.text(i, 0, v, horizontalalignment = 'center', size = 14, color = 'w', fontweight = 'bold')
        
        fig.savefig(os.path.join(save_path, 'gender.png'))

        # Age
        sns.displot(df, x="age", stat="density")
        plt.savefig(os.path.join(save_path, 'age.png'))

        # Group Age
        age_0 = df[df['age'] < 30].value_counts().sum()
        age_1 = df[(30 <= df['age']) & (df['age'] < 60)].value_counts().sum()
        age_2 = df[df['age'] >= 60].value_counts().sum()

        group_age = np.array([age_0, age_1, age_2])
        group = ['~ 29', '30 ~ 59', '60 ~']

        fig, ax = plt.subplots(1, 1, figsize=(12,6))

        ax.bar(group, group_age)
        ax.set_title('Group Age', fontsize= 14)
        ax.set_xticklabels(['~ 29', '30 ~ 59', '60 ~'], fontsize=14)
        fig.savefig(os.path.join(save_path, 'group_age.png'))

        # Age & Gender
        sns.displot(df, x="age", hue="gender", stat="density")
        plt.savefig(os.path.join(save_path, 'age_n_gender.png'))

        sns.boxplot(x='gender', y='age', data=df)
        plt.savefig(os.path.join(save_path, 'gender_n_age.png'))

    if xy:
        # Gray Scale인 경우 Histogram
        img_id = df.iloc[500].path
        ext = get_ext(cfg.img_dir, img_id)

        plt.figure()
        plt.subplot(111)

        for class_id in num2class:
            img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, class_id+ext)).convert('L'))
            histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
            sns.lineplot(data=histogram)

        plt.legend(num2class)
        plt.title('Class Grayscale Histogram Plot', fontsize=15)
        plt.savefig(os.path.join(save_path, 'RGB_Y.png'))

        # Mask를 쓴 이미지는 평균을 내서 확인
        plt.figure()
        plt.subplot(111)

        img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, 'incorrect_mask'+ext)).convert('L'))
        histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
        sns.lineplot(data=histogram)

        img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, 'normal'+ext)).convert('L'))
        histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
        sns.lineplot(data=histogram, color='hotpink')

        histograms = []
        for i in range(1, 6):
            img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, num2class[i]+ext)).convert('L'))
            histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
            histograms.append(histogram)
        sns.lineplot(data=np.mean(histograms, axis=0))

        plt.legend(['incorrect_mask', 'normal', 'mask average'])
        plt.title('Class Grayscale Histogram Plot', fontsize=15)
        plt.savefig(os.path.join(save_path, 'RGB_Y_mask_mean.png'))

        # 마스크를 쓰지 않은 경우의 R, G, B의 histogram
        plt.figure()
        plt.subplot(111)

        img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, 'normal'+ext)))
        colormap = ['red', 'green', 'blue']
        for i in range(3):
            histogram, bin_edges = np.histogram(img[..., i], bins=256, range=(0, 255))
            sns.lineplot(data=histogram, color=colormap[i])

        plt.legend()
        plt.title('RGB Histogram Plot - Normal', fontsize=15)
        plt.savefig(os.path.join(save_path, 'RGB_Y_no_mask.png'))

    if not_detection_img:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        imgs = []
        bboxes = []
        not_found_idx = []
        img_id = df.iloc[504].path
        ext = get_ext(cfg.img_dir, img_id)
        for i, class_id in enumerate(num2class):
            img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, class_id+ext)))
            bbox = face_cascade.detectMultiScale(img)
            imgs.append(img)
            if len(bbox) != 0:
                bboxes.append(bbox.max(axis=0))
            else:
                not_found_idx.append(i)
                print(f'{class_id} not found face')
        imgs = np.array(imgs)
        bboxes = np.array(bboxes)

        fig, axes = plt.subplots(1, len(not_found_idx), sharex=True, sharey=True, figsize=(12, 6))
        for i, j in enumerate(range(len(not_found_idx))):
            axes[i].imshow(imgs[j])
            axes[i].set_title(f'{num2class[j]}')
        fig.savefig(os.path.join(save_path, './EDA_test_img/not_found_face.png'))

    if n_not_detection:
        not_found = count_face_detection(cfg.img_dir, df.path.values)
        x = ['incorrect', 'mask1', 'mask2', 'mask3',
             'mask4', 'mask5', 'normal']
        not_found_cnt = {k:len(v) for k, v in not_found.items()}
        y = np.array(list(not_found_cnt.values()))
        colors = ['#87CEFA', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#F08080']
        plt.bar(x, y, color=colors)
        for idx, val in enumerate(y):
            plt.text(x=idx, y=val, s=val,
                     va='bottom', ha='center',
                     fontsize=11, fontweight='semibold'
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, './EDA_test_img/all_not_found_face.png'))
        

        with open('not_found.pickle', 'wb') as f:
            pickle.dump(not_found, f)

    if no_mask_not_detection:
        with open("not_found.pickle", 'rb') as f:
            not_found = pickle.load(f)
        for id in range(len(not_found['normal'])):
            img = np.array(Image.open(not_found['normal'][id]))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(save_path, f'no_mask_not_detection{id+1}.png'))
            print((id+1), not_found['normal'][id])

    if no_mask_noise:
        no_mask_img = [2399, 2400, 1912, 764]
        fig, axes = plt.subplots(1, 4, figsize=(8,4))
        for idx, id in enumerate(no_mask_img):
            img_id = df.iloc[id].path
            gen = df.iloc[id].gender
            age = df.iloc[id].age
            exts = get_ext(cfg.img_dir, img_id)
            for ext in exts:
                try:
                    img_path = os.path.join(cfg.img_dir, img_id, 'normal' + ext)
                    img = np.array(Image.open(img_path))
                    print(img_path)
                    break
                except:
                    continue
            axes[idx].imshow(img)
            axes[idx].set_title(f'{gen} {age}')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'no_mask.png'))

    if mask_noise:
        plot_mask_images(cfg.img_dir, 2399, save_path)

    # 결과 해석을 어떻게 해야하지
    if tSNE:
        # PCA
        n_imgs = 100

        imgs = []
        for img_id in df.path.values[:n_imgs]:
            exts = get_ext(cfg.img_dir, img_id)
            for class_id in num2class:
                for ext in exts:
                    try:
                        img_path = os.path.join(cfg.img_dir, img_id, class_id + ext)
                        img = np.array(Image.open(img_path).convert('L'))
                        break
                    except:
                        continue
                imgs.append(img)
        imgs = np.array(imgs)
        n_samples, h, w = imgs.shape

        imgs = np.reshape(imgs, (n_samples, h*w))

        n_components = 30

        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized',
                whiten=True).fit(imgs)
        print(f"pca is fitted in {time() - t0:.0f}s")
        print(f'Explained variation per principal component: \n{pca.explained_variance_ratio_}')

        eigenfaces = pca.components_.reshape((n_components, h, w))
        img_pca = pca.transform(imgs)

        pca_df = pd.DataFrame(img_pca, columns=[str(col) for col in range(n_components)])
        pca_df['class_id'] = [num2class[n % len(num2class)] for n in range(n_samples)]
        pca_df['class_id'] = pca_df['class_id'].map(lambda x: x if x in ['incorrect_mask', 'normal'] else 'mask')

        # tSNE
        time_start = time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(img_pca)
        print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))

        pca_df['tsne-2d-one'] = tsne_results[:,0]
        pca_df['tsne-2d-two'] = tsne_results[:,1]
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="class_id",
            palette=sns.color_palette("Set2", 3),
            data=pca_df,
            legend="full",
            alpha=0.8
        )
        plt.savefig(os.path.join(save_path, 'tsne.png'))

    if make_csv:
        data = []
        for idx in tqdm(range(len(df))):
            img_id = df.iloc[idx].path
            gen = df.iloc[idx].gender
            age = df.iloc[idx].age

            exts = get_ext(cfg.img_dir, img_id)
            for class_name in num2class:
                ans = ''
                for ext in exts:
                    try:
                        img_path = os.path.join(cfg.img_dir, img_id, class_name + ext)
                        img = np.array(Image.open(img_path))
                        break
                    except:
                        continue
                if class_name == 'incorrect_mask':
                    ans += '1'
                elif class_name == 'normal':
                    ans += '2'
                else:
                    ans += '0'
                
                if gen == 'male':
                    ans += '0'
                else:
                    ans += '1'

                if age < 30:
                    ans += '0'
                elif age >= 60:
                    ans += '2'
                else:
                    ans += '1'

                data.append([img_path, ans])

        result_df = pd.DataFrame(data, columns=['ImageID', 'ans'])
        result_df.to_csv(save_csv_path, index=False)

    if test_make_csv:
        result_df = pd.read_csv(save_csv_path)
        print(result_df.head())
        print(result_df.iloc[0].ImageID)
        print(result_df.iloc[0].ans)
        
        print(result_df['ans'].value_counts())

    if class_distribution:
        info_df = pd.read_csv('../input/data/train/train_info.csv')
        label = info_df['ans'].value_counts()
        label_dict = sorted(dict(label).items(), key=lambda x:x[0])

        x = [str(key) for key, val in label_dict]
        y = [val for key, val in label_dict]
        colors = ['#87CEFA'] * 18
        plt.bar(x, y, color=colors)
        for idx, val in enumerate(y):
            plt.text(x=idx, y=val, s=val,
                     va='bottom', ha='center',
                     fontsize=11, fontweight='semibold'
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'train_info_ans2.png'))

    if make_split_csv:
        dtype_dict = {'ImageID':'str', 'ans':'str'}
        df = pd.read_csv('../input/data/train/train_info.csv', dtype=dtype_dict)
        class_list = ['000', '001', '002', '010', '011', '012',
                      '100', '101', '102', '110', '111', '112',
                      '200', '201', '202', '210', '211', '212']
        np.random.seed(223)
        new_df = pd.DataFrame(columns=['ImageID', 'ans', 'type'])
        for class_num in class_list:
            class_df = df[df['ans']==class_num]
            shuffle_df = sklearn.utils.shuffle(class_df, random_state=223)
            train_df = shuffle_df[20:].reset_index(drop=True)
            val_df = shuffle_df[:20].reset_index(drop=True)
            train_df['type'] = 'train'
            val_df['type'] = 'val'
            new_df = pd.concat([new_df, train_df, val_df], ignore_index=True)

        new_df = sklearn.utils.shuffle(new_df, random_state=223)
        
        new_df.to_csv('../input/data/train/train_info2.csv', index=False)

    if make_limit_csv:
        dtype_dict = {'ImageID':'str', 'ans':'str'}
        df = pd.read_csv('../input/data/train/train_info.csv', dtype=dtype_dict)
        class_list = ['000', '001', '002', '010', '011', '012',
                      '100', '101', '102', '110', '111', '112',
                      '200', '201', '202', '210', '211', '212']
        np.random.seed(223)
        new_df = pd.DataFrame(columns=['ImageID', 'ans', 'type'])
        for class_num in class_list:
            class_df = df[df['ans']==class_num]
            shuffle_df = sklearn.utils.shuffle(class_df, random_state=223)
            train_df = shuffle_df[20:520].reset_index(drop=True)
            val_df = shuffle_df[:20].reset_index(drop=True)
            train_df['type'] = 'train'
            val_df['type'] = 'val'
            new_df = pd.concat([new_df, train_df, val_df], ignore_index=True)

        new_df = sklearn.utils.shuffle(new_df, random_state=223)
        
        new_df.to_csv('../input/data/train/train_info4.csv', index=False)

    if transforms_test:
        save_path = './transform_exp'
        img_path = '../input/data/train/images/004414_female_Asian_20/normal.jpg'
        img = Image.open(img_path)
        img.save(os.path.join(save_path, 'female_20.png'))

        transform_list = {'Crop_334_334' : transforms.CenterCrop((384, 384)),
                          'Crop_256_256' : transforms.CenterCrop((256, 256)),
                          'Rotation_10' : transforms.RandomRotation((10, 10)),
                          'Rotation_30' : transforms.RandomRotation((30, 30)),
                          'Horizontal' : transforms.RandomHorizontalFlip(1),
                          'Brightness_0.2' : transforms.ColorJitter(brightness=(0.2,0.2)),
                          'Brightness_0.5' : transforms.ColorJitter(brightness=(0.5,0.5)),
                          'Brightness_0.8' : transforms.ColorJitter(brightness=(0.8,0.8)),
                          'Brightness_1.0' : transforms.ColorJitter(brightness=(1.0,1.0)),
                          'Brightness_1.2' : transforms.ColorJitter(brightness=(1.2,1.2)),
                          'Brightness_1.5' : transforms.ColorJitter(brightness=(1.5,1.5)),
                          'Brightness_1.8' : transforms.ColorJitter(brightness=(1.8,1.8)),
                          'Brightness_2.0' : transforms.ColorJitter(brightness=(2.0,2.0)),
                          'Contrast_0.2' : transforms.ColorJitter(contrast=(0.2,0.2)),
                          'Contrast_0.5' : transforms.ColorJitter(contrast=(0.5,0.5)),
                          'Contrast_0.8' : transforms.ColorJitter(contrast=(0.8,0.8)),
                          'Contrast_1.0' : transforms.ColorJitter(contrast=(1.0,1.0)),
                          'Contrast_1.2' : transforms.ColorJitter(contrast=(1.2,1.2)),
                          'Contrast_1.5' : transforms.ColorJitter(contrast=(1.5,1.5)),
                          'Contrast_1.8' : transforms.ColorJitter(contrast=(1.8,1.8)),
                          'Contrast_2.0' : transforms.ColorJitter(contrast=(2.0,2.0)),
                          'Saturation_0.2' : transforms.ColorJitter(saturation=(0.2,0.2)),
                          'Saturation_0.5' : transforms.ColorJitter(saturation=(0.5,0.5)),
                          'Saturation_0.8' : transforms.ColorJitter(saturation=(0.8,0.8)),
                          'Saturation_1.0' : transforms.ColorJitter(saturation=(1.0,1.0)),
                          'Saturation_1.2' : transforms.ColorJitter(saturation=(1.2,1.2)),
                          'Saturation_1.5' : transforms.ColorJitter(saturation=(1.5,1.5)),
                          'Saturation_1.8' : transforms.ColorJitter(saturation=(1.8,1.8)),
                          'Saturation_2.0' : transforms.ColorJitter(saturation=(2.0,2.0)),
                          'Hue_minus_0.5' : transforms.ColorJitter(hue=(-0.5,-0.5)),
                          'Hue_minus_0.3' : transforms.ColorJitter(hue=(-0.3,-0.3)),
                          'Hue_minus_0.1' : transforms.ColorJitter(hue=(-0.1,-0.1)),
                          'Hue_0.1' : transforms.ColorJitter(hue=(0.1,0.1)),
                          'Hue_0.3' : transforms.ColorJitter(hue=(0.3,0.3)),
                          'Hue_0.5' : transforms.ColorJitter(hue=(0.5,0.5)),
                          'RandomAffine_degrees' : transforms.RandomAffine(degrees=(30,30),
                                                                           translate=(0,0),
                                                                           shear=(0,0,0,0)),
                          'RandomAffine_translate' : transforms.RandomAffine(degrees=(0,0),
                                                                             translate=(0.1,0.1),
                                                                             shear=(0,0,0,0)),
                          'RandomAffine_shear' : transforms.RandomAffine(degrees=(0,0),
                                                                         translate=(0,0),
                                                                         shear=(10,10,10,10))}

        
        save_path = './transform_exp'

        for key in transform_list.keys():
            trans_img = transform_list[key](img)
            trans_img.save(os.path.join(save_path, f'{key}_female_20.png'))

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225)),
                                        transforms.ToPILImage()])
        trans_img = transform(img)
        trans_img.save(os.path.join(save_path, 'Normalize_ImageNet_female_20.png'))

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5601, 0.5241, 0.5014),
                                                             std=(0.2331, 0.2430, 0.2456)),
                                        transforms.ToPILImage()])
        trans_img = transform(img)
        trans_img.save(os.path.join(save_path, 'Normalize_mask_female_20.png'))
        


    # dtype_dict = {'ImageID':'str', 'ans':'str'}
    # new_df = pd.read_csv('../input/data/train/train_info4.csv', dtype=dtype_dict)
    # new_df = new_df[new_df['type']=='train']
    # class_list = ['000', '001', '002', '010', '011', '012',
    #               '100', '101', '102', '110', '111', '112',
    #               '200', '201', '202', '210', '211', '212']

    # class_dict = dict(new_df['ans'].value_counts())

    # count = {}
    # for key, val in class_dict.items():
    #     count[str(class_list.index(key))] = int(val)
    # count_json = json.dumps(count)
    # with open('./train_cnt.json','w') as f:
    #     json.dump(count_json, f)

    


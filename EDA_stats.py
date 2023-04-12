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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
    make_csv = True
    test_make_csv = True
    tSNE = False
    save_csv_path = './input/data/train/train_info.csv'

    class cfg:
        data_dir = './input/data/train'
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
                ans = 0
                for ext in exts:
                    try:
                        img_path = os.path.join(cfg.img_dir, img_id, class_name + ext)
                        img = np.array(Image.open(img_path))
                        break
                    except:
                        continue
                if class_name == 'incorrect_mask':
                    if gen == 'male':
                        if age < 30: ans = 6
                        elif age >= 60: ans = 8
                        else: ans = 7
                    else:
                        if age < 30: ans = 9
                        elif age >= 60: ans = 11
                        else: ans = 10
                elif class_name == 'normal':
                    if gen == 'male':
                        if age < 30: ans = 12
                        elif age >= 60: ans = 14
                        else: ans = 13
                    else:
                        if age < 30: ans = 15
                        elif age >= 60: ans = 17
                        else: ans = 16
                else:
                    if gen == 'male':
                        if age < 30: ans = 0
                        elif age >= 60: ans = 2
                        else: ans = 1
                    else:
                        if age < 30: ans = 3
                        elif age >= 60: ans = 5
                        else: ans = 4
                data.append([img_path, ans])

        result_df = pd.DataFrame(data, columns=['ImageID', 'ans'])
        result_df.to_csv(save_csv_path, index=False)

    if test_make_csv:
        result_df = pd.read_csv(save_csv_path)
        print(result_df.head())
        print(result_df.iloc[0].ImageID)
        print(result_df.iloc[0].ans)
        
        print(result_df['ans'].value_counts())

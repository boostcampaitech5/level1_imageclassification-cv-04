import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def weighted_sampler(dataset, num_classes):
    print(dataset.get_df())
    class_counts = np.zeros(num_classes)
    labels = []
    for _, label in dataset:
        class_counts[label] += 1
        labels.append(label)
    num_samples = len(dataset)
    
    class_weights = num_samples/class_counts
    print(class_counts,class_weights)
    # 해당 데이터에 해당하는 class 가중치
    weights = [class_weights[labels[i]] for i in range(num_samples)]
    sampler = WeightedRandomSampler(torch.FloatTensor(weights), num_samples)

    return sampler
from dataloader.dataset import ClassificationDataset
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
transform = transforms.Compose([transforms.ToTensor()])

dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                transform=transform)
print(dataset.df.columns)

train_set, val_set = train_test_split(dataset.df['ImageID'],
                                          dataset.df['ans'],
                                          train_size=0.8,
                                          stratift=dataset.df['ans'])

print(train_set)
import pandas as pd

TRAIN_IMG_DIR = "/opt/ml/input/data/eval/"
df = pd.read_csv(TRAIN_IMG_DIR + "info.csv")

arr = ['6769d4d2118bb4855919e72344b3b0298ca53ac8.jpg', '7296e8c80bf2ec11baa64361d33b908ebfb78820.jpg', '95a7739a34fc0719231534e42851ee1ffc6ec288.jpg', '5e8503bf2c9e2e927488365c273713988fa4596f.jpg', 'a76263c8f8133eac525d6ecfc6b96393a85bb458.jpg', 'c8f54bbf55d0064434cc706b072c4cdb7d7c647f.jpg', '2d6417bc25cad180c2d52ee864a96d57124d1b2e.jpg', 'e9d72ff56cebac258d36d6562c61220bbbd28ea6.jpg', '83c482b72f101e81e2458b480f62f792b9c6db59.jpg', 'dfdc964822c0bb76bd27ca829bfbb8bcdaa42c42.jpg']

for i in arr:
    df.loc[df["ImageID"] == arr] = 1

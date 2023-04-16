import os
import cv2


image_path = './input/data/eval/images'

image_list = os.listdir(image_path)

for img in image_list:
    if img[0] != ".":
        img_path = os.path.join(image_path,img)
        image = cv2.imread(img_path)
        # center = image.shape
        # x = center[1]/2 - w/2
        # y = center[0]/2 - h/2

        # crop_img = img[int(y):int(y+h), int(x):int(x+w)]
        
        cv2.imshow('a',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
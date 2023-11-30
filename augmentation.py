import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# finding avg size
def avg_img_size(folder_path):
    sizes = []
    # print(os.listdir(folder_path))
    for filename in os.listdir(folder_path):
        
        if filename.endswith(('jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                sizes.append(img.size)

    sizes = np.array(sizes)
    avg_size = sizes.mean(axis=0)
    return avg_size


avg = avg_img_size("data/test")
print(f"Average image size (test): {avg}")

avg1 =  avg_img_size("data/train")
print(f"Average image size (train): {avg1}")

class FocusedCrop(torch.nn.Module):
  
    def __init__(self, crop_percentage):
        super(FocusedCrop, self).__init__()
        self.crop_percentage = crop_percentage

    def forward(self, img):
        
        width, height = img.size
        left = width * (1 - self.crop_percentage) / 2
        top = height * (1 - self.crop_percentage) / 2
        right = width * (1 + self.crop_percentage) / 2
        bottom = height * (1 + self.crop_percentage) / 2
        # Built in img crop
        return img.crop((left, top, right, bottom))
    

# testing

crop_percentage = 0.9 #test 80 %
crop_transform = transforms.Compose([FocusedCrop(crop_percentage),])

def process_image(path, transform): 
    with Image.open(path) as img:
        img.show()
        cropped_img = transform(img)
        cropped_img.show()

process_image("data/train/4021.jpg", crop_transform)

'''

Notes:
Average image size (test): [672.68848039 465.21397059]
Average image size (train): [667.46674728 461.86674728]

It might be best to transform everything to a square. Most classification models work on squared images. 

Another option is to use a sliding window approach: 
- Basically "slide" a box around image and classify each image crop inside of a box
- This is how "real world" classifiers work, but cropping to squares might work just fine!

We can test with both - it is easy to change as it is just part of a transform

'''
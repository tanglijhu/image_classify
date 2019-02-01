import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, tensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import utils
import os, sys

#Command Line Arguments
ap = argparse.ArgumentParser(description='predict.py')
ap.add_argument('input_img', nargs=None, type=str, action="store") # './flowers/test/1/image_06760.jpg'
ap.add_argument('--checkpoint', nargs=None, type=str, action="store", default='./checkpoint.pth')
ap.add_argument('--top_k', dest="top_k", type=int, action="store", default=5)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

pa = ap.parse_args()
path_image = pa.input_img
input_img = pa.input_img
path = pa.checkpoint
outputs_num = pa.top_k
cat_name = pa.category_names


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

print("******************* Prediction Starts! ***************************")


#train_dataset, validate_dataset, test_dataset, train_dataloader, validate_dataloader, test_dataloader = utils.load_data(path_image)

model = utils.load_checkpoint(path)

with open(cat_name, 'r') as json_file:
    cat_to_name = json.load(json_file)

image_index = splitall(path_image)[3]    
correct_class = cat_to_name[image_index]

top_prob, top_classes = utils.predict(path_image, model, outputs_num)

label = top_classes[0]

labels = []

for class_idx in top_classes:
    labels.append(cat_to_name[class_idx])
print(labels)

i=0
while i < outputs_num:
    print("{} with a probability of {}".format(labels[i], top_prob[i]))
    i += 1

print(f'Correct classification: {correct_class}')
print(f'Correct prediction: {correct_class == cat_to_name[label]}')

print("******************* Prediction is Done! ***************************")
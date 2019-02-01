import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn, tensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import utils

ap = argparse.ArgumentParser(description='train.py')

# Command Line ardguments
ap.add_argument('--data_dir', nargs=None, action="store", default="./flowers")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--check_steps',dest='check_steps',action='store',type=int, default=50)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.0015)
ap.add_argument('--dropout', dest = "dropout", action = "store", type=float, default=0.15)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
ap.add_argument('--gpu', dest="gpu", action="store", type=str, default="gpu")
ap.add_argument('--arch', dest="arch", action="store", type=str, default="vgg16")
ap.add_argument('--hidden_layer', dest="hidden_layer", action="store", type=int, default=16725)

pa = ap.parse_args()
place = pa.data_dir
path = pa.save_dir
check_steps = pa.check_steps
lr = pa.learning_rate
dropout = pa.dropout
epochs = pa.epochs
power = pa.gpu
arch = pa.arch
hidden_size = pa.hidden_layer

train_dataset, validate_dataset, test_dataset, train_dataloader, validate_dataloader, test_dataloader = utils.load_data(place)

model, classifier, criterion, optimizer = utils.nn_setup(arch, dropout, hidden_size, lr, power)

utils.train_network(model, criterion, optimizer, train_dataloader, validate_dataloader, epochs, check_steps, lr, power)

utils.save_checkpoint(model, classifier, train_dataset, path)

print("All Set and Done! The Model is trained") 

# Import packages
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json

# Model options
arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

# Define functions  
def load_data(place  = "./flowers" ):
    '''
    *** Load image data to be processed 
    Arguments : file path for image data
    Returns : dataset for training data and dataloaders for the train, validation and test
    '''
    # File paths
    data_dir = place
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Apply transfomations to the dataset 
    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    image_size = 224
    batch_size = 64
    data_transforms = {
        "training": transforms.Compose([transforms.RandomHorizontalFlip(p=0.20),
                                    transforms.RandomRotation(30),
                                    transforms.RandomGrayscale(p=0.015),
                                    transforms.RandomResizedCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(expected_means, expected_std)]),
        "validation": transforms.Compose([transforms.Resize(image_size + 2),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(expected_means, expected_std)]),
        "testing": transforms.Compose([transforms.Resize(image_size + 2),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(expected_means, expected_std)])
    }
    
    # Applying data transformations to datasets
    image_datasets = {
        "training": datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "testing": datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
    }  
    
    # Load datasets after transformations
    dataloaders = {
        "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=batch_size, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=batch_size),
        "testing": torch.utils.data.DataLoader(image_datasets["testing"], batch_size=batch_size)
    }
    
    train_dataset = image_datasets['training']
    validate_dataset = image_datasets['validation']  
    test_dataset = image_datasets['testing']   
    train_dataloader = dataloaders["training"]
    validate_dataloader = dataloaders["validation"]
    test_dataloader = dataloaders["testing"]
    
    return train_dataset, validate_dataset, test_dataset, train_dataloader, validate_dataloader, test_dataloader

def nn_setup(structure='vgg16', dropout=0.15, hidden_layer = 16725, lr = 0.0015, power = 'gpu'):
    '''
    *** Set up the training model 
    Arguments: training network (i.e., vgg16), the hyperparameters for the network (i.e., dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, and the criterion and the optimizer 
    '''
    # Choose the arch option as a pre-trained network
    if structure == 'vgg16':
        nn_model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        nn_model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        nn_model = models.alexnet(pretrained = True)
    else:
        print("This is not a valid model. Do you want to choose vgg16, densenet121, or alexnet?")
    
    # Figure out the input_size from the classifier of the pre-trained model, vgg16, debsenet121, or alexnet
    #input_size = nn_model.classifier[0].in_features
    input_size = arch[structure]
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Figure out the output_size size
    output_size = len(cat_to_name)
    
    # Setup the hidden layer(s) size
    hidden_size = int(round(input_size / 3 * 2))

    # Stop backpropigation on parameters
    for param in nn_model.parameters():
        param.requires_grad = False 
    
    # Create nn.Module with Sequential using an OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_size)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p=dropout)),
        ('output', nn.Linear(hidden_size, output_size)),
        # Use LogSoftmax as NLLLoss criterion
        ('softmax', nn.LogSoftmax(dim=1))]))
    
    # Replace the classifier of the current training network with the new classifier from the pre-trained network
    nn_model.classifier = classifier
    
    # Use NLLLoss as criterion
    criterion = nn.NLLLoss()
    
    # Use optimizer Adam for Optimization
    optimizer = optim.Adam(nn_model.classifier.parameters(), lr)
    
    # Check the mode, cpu or gpu
    if torch.cuda.is_available() and power == 'gpu':
        nn_model.cuda()
   
    return nn_model, classifier, criterion, optimizer

def train_network(model, criterion, optimizer, train_dataloader, validate_dataloader, epochs=10, check_steps=50, lr=0.0015, power = 'gpu'):   
    '''
    *** Train the model 
    Arguments: model, criterion, optimizer, number of epochs, check steps, learning rate, training dataloader, validation dataloader
    Returns: none
    '''
    # Setup hyperparameters
    epochs = epochs
    lr = lr
    check_steps = check_steps
    
    print("--------------Training is starting------------- ")
    
    # Set gradients of all parameters to zero
    nn_model = model
    nn_model.zero_grad()

    # Move model to perferred device.
    if torch.cuda.is_available() and power == 'gpu':
        nn_model.cuda()
    
    # Figure out the size of training set and the size of total images of validation set
    traingset_no = len(train_dataloader.batch_sampler)
    total_training_images = len(train_dataloader.batch_sampler) * train_dataloader.batch_size
    total_validation_images = len(validate_dataloader.batch_sampler) * validate_dataloader.batch_size

    print(f'Using {traingset_no} batches with each bacth of {train_dataloader.batch_size}.')

    for e in range(epochs):    
        nn_model.train()   # Put the model into train mode      
        e_loss = 0
        prev_check = 0
        total = 0
        correct = 0
        
        print(f'\nEpoch {e+1} of {epochs}\n************************************')
      
        for i, (images, labels) in enumerate(train_dataloader):
            # Move images and labeles preferred device
            if torch.cuda.is_available() and power == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
 
            # Set gradients of all parameters to zero. 
            optimizer.zero_grad()

            # Propigate forward and then backward 
            outputs = nn_model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Keep a running total of loss for this epoch
            e_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            i_check = (i + 1)
            if i_check % check_steps == 0:

                valid_loss = 0
                valid_correct = 0
                valid_total = 0
                nn_model.eval()

                with torch.no_grad():
                    for ii, (images, labels) in enumerate(validate_dataloader):
                        # Move images and labeles perferred device
                        if torch.cuda.is_available() and power == 'gpu':
                            images, labels = images.to('cuda'), labels.to('cuda')
                        batch_outputs = nn_model.forward(images)            
                        batch_loss = criterion(batch_outputs, labels)

                        # Keep a running total of loss for this epoch
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        _, predicted = torch.max(batch_outputs.data, 1)
                        valid_total += labels.size(0)
                        valid_correct += (predicted == labels).sum().item()

                #print(f"\n\tValidating for epoch {e+1}")
                avg_valid_loss = f'avg. validation loss: {valid_loss / valid_total:.4f}'
                correct_perc = 0
                if valid_correct > 0:
                    correct_perc = (100 * valid_correct // valid_total)

                print(f'Showing average loss and accuracy at {i_check} batches:\n'
                      f"Train loss: {e_loss/check_steps:.3f}.. "
                      f'Train accuracy: {(correct/total) * 100:.2f}%..'
                      f"Validate loss: {valid_loss/(ii+1):.3f}.. "
                      f"Validate accuracy: {valid_correct/valid_total:.3f}")
                e_loss = 0
    
    print("-------------- Training Done -----------------------")
  

def save_checkpoint(model, arch, classifier, train_dataset, path='checkpoint.pth'):
    '''
    *** Save the parameters of the trained model
    Arguments: saving path and the hyperparameters of the network
    Returns: none
    '''
    nn_model = model   
    nn_model.cpu
    nn_model.class_to_idx = train_dataset.class_to_idx
    nn_model.arch = arch
    
    model_state = {
    'classifier': nn_model.classifier,
    'state_dict': nn_model.state_dict(),
    'class_to_idx': nn_model.class_to_idx,
    'arch': nn_model.arch,
}
    torch.save(model_state, path)

def load_checkpoint(filepath='checkpoint.pth'):
    '''
    *** Load weights for CPU model
    Arguments: loading path and the hyperparameters of the trained network
    Returns: trained model
    '''
    model_state = torch.load(filepath)
    model = models.__dict__[model_state['arch']](pretrained=True)
    model.classifier = model_state["classifier"]
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']
    return model


def process_image(image):
    ''' 
    *** Process an image 
    Arguments: scales, crops, and normalizes a PIL image for a PyTorch model
    Returns: an torch.Tensor
    '''    
    # Process a PIL image for use in a PyTorch model
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
    
    # Open image       
    image = Image.open(image).convert("RGB")
    
    # Process image
    current_width, current_height = image.size
    if current_width < current_height:
        new_height = int(current_height * 256 /current_width)
        image = image.resize((256,new_height))
    else:
        new_width = int(current_width * 256 /current_height)
        image = image.resize((new_width,256))
    
    precrop_width, precrop_height = image.size
    left = (precrop_width - 224) / 2
    top = (precrop_height - 224) / 2
    right = (precrop_width + 224) / 2
    bottom = (precrop_height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = test_transforms(image)

    return pil_image


def predict(image_path, model, power, topk=5):
    ''' 
    *** Predict the class (or classes) of an image using a trained deep learning model
    Arguments: get the processed image data from the image path
    Returns: a prediction
    '''    
    # Chaneg to evaluation mode to predict its class for an image file
    model.eval()
    
    # Change to the CPU mode
    model.cpu()
    
    # Move model to perferred device.
    device = torch.device("cuda:0" if (torch.cuda.is_available() and power == 'gpu') else "cpu")
    #model = model.to(device)
    print(device)
    
    # Load image as torch.Tensor
    image = process_image(image_path)
    
    # Unsqueeze the tensor so it returns a new tensor with a dimension of size one
    image = image.unsqueeze(0)
    
    
    # Stop gradient calculation 
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes
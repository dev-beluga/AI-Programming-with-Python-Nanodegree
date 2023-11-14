import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

def load_data(dataset_path):
    """
    Load the data
    The dataset is split into three parts:
        training, 
        validation, and 
        testing.
    """
    data_dir = dataset_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define  transforms for the training, validation, and testing datasets
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)


    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    image_datasets = [train_data, valid_data, test_data]
    dataloaders = [trainloader, validloader, testloader]

    return trainloader,validloader, testloader, train_data
    
    
def process_image(image_path):
    """
    Image Preprocessing
    This function preprocesses the image so it can be used as input for the model.
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    Returns an Numpy array.
    """
    # Resize the image to a size of (256, 256)
    img = Image.open(image_path)
    img = img.resize((256,256))

    # Calculate the coordinates for center cropping
    left = (256 - 224) / 2.0
    up = (256 - 224) / 2.0
    right = (256 - 224) / 2.0
    bottom = (256 - 224) / 2.0

    img_cropped = img.crop((left,up,256-right,256-bottom))
    # Convert the PIL image to a NumPy array
    np_image = np.array(img_cropped)
    img = np_image / 255.0

    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    img = (img - norm_mean) / norm_std
    # Transpose the NumPy array to reorder the dimensions
    img = img.transpose((2, 0, 1))  # Change (height, width, channels) to (channels, height, width)
    return img

def load_checkpoint(filepath):
    '''loads the check point'''
    checkpoint = torch.load(filepath)
    model= checkpoint['model']        
    return model

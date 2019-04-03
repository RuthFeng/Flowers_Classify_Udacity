
import torch
from torchvision import datasets,transforms,models
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import json

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    model = models.__dict__[chpt['arch']](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = chpt['class_to_idx']
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image    
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img  

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, trained_modelfile,top_num,category_namesfile,gpu_on):
    
    #Load trained modelfile
    model = load_model(trained_modelfile)
    
    # Process image
    img = process_image(image_path)


    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    if gpu_on=='Y': 
        if torch.cuda.is_available(): 
            print('***gpu_on =Y*** ')
            model.to('cuda')      
            image_tensor=image_tensor.to('cuda')
        else:
            print('Initiate GPU Failed, switch to CPU mode')
    
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    
    with open(category_namesfile, 'r') as f:
        cat_to_name = json.load(f)

    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

def get_command_line_arguments():

    parser = argparse.ArgumentParser(description='Parse image_path trained_modelfile  --top_num --category_namesfile --gpu_on from the command line')
    parser.add_argument('image_path', default='flowers/valid/14/image_06082.jpg', type=str, help="gives the image_path include file name,default flowers/valid/14/image_06082.jpg")
    
    parser.add_argument('trained_modelfile', default='checkpoint.pth', type=str, help="gives the tained modle\'s save path include file name,default checkpoint.pth")
    parser.add_argument('--top_num', default=5, type=int, help="gives the top n probablity, default 5")
    parser.add_argument('--category_namesfile', default='cat_to_name.json', type=str, help="gives the category_namesfile,default cat_to_name.json")
    parser.add_argument('--gpu_on', default='Y', help="turn on\(Y\) gpu or not\(N\) , default Y")
    args = parser.parse_args()
    return args
    
def main ():
     args = get_command_line_arguments()
     top_probs, top_labels, top_flowers= predict(args.image_path,args.trained_modelfile,args.top_num,args.category_namesfile,args.gpu_on)

     print(args.image_path,'is category:')
     print(list(zip(top_flowers,top_probs)))

# Call to main function to run the program
if __name__ == "__main__":
    main()
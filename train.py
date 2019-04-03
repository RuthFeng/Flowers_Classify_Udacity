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
import copy

def load_data (root_dir,Resize=256,crop=224):    
    train_dir=root_dir + '/train'
    test_dir=root_dir + '/test'
    valid_dir=root_dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(Resize),
                                       transforms.RandomResizedCrop(crop),                                       
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(Resize),
                                      transforms.CenterCrop(crop),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    valid_data = datasets.ImageFolder(valid_dir,transform=test_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False)

    return train_data,train_loader,test_data,test_loader,valid_data,valid_loader

def initial_model(arch='vgg16',input_layer=25088, hidden_layer=4096,output_layer=102):
    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_layer,hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer,output_layer)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return model

def train (root_dir,save_dir='checkpoint.pth',arch='vgg16',learning_rate=0.003,input_layer=25088, hidden_layer=4096,output_layer=102,epochs=4,gpu_on='Y'):
   
    train_data,train_loader,test_data,test_loader,valid_data,valid_loader=load_data (root_dir,256,224)
    
    model = initial_model(arch,input_layer, hidden_layer,output_layer)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    learn_rate= learning_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    epochs = epochs
    print_every = 40
    steps = 0

    # change to cuda
    gpu_on=gpu_on
    if gpu_on=='Y': 
        if torch.cuda.is_available():     
            print('***gpu_on =Y*** ')
            model.to('cuda')
        else:
            print('Initiate GPU Failed, switch to CPU mode')
        

    print('***Training Begin***')
    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1         
        
            if gpu_on=='Y':
                if torch.cuda.is_available():              
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                else:
                    print('Initiate GPU Failed, switch to CPU mode')              
        
            optimizer.zero_grad()
                  
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()       
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()     
                # TODO: Do validation on the valid set

                correct = 0
                total = 0

                with torch.no_grad():
                    for ii, (images, labels) in enumerate(test_loader):
                        #images, labels = data
                        images, labels = images.to('cuda'), labels.to('cuda') 
        
                        outputs = model(images)
        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Accuracy: {:.3f}".format(correct / total))
            
                running_loss = 0
            
                # Make sure dropout is on for training
                model.train()
                
    print('***Training End***\n')

    print('***Test begin***\n')
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for ii, (images, labels) in enumerate(test_loader):
        #images, labels = data
            if gpu_on=='Y':
                if torch.cuda.is_available():              
                    images, labels = images.to('cuda'), labels.to('cuda') 
                else:
                    print('Initiate GPU Failed, switch to CPU mode')               
        
            outputs = model(images)
        
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('***Test End. Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    torch.save({'arch': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx,
             'accuracy': correct / total}, 
             save_dir)
    print('The trained model is saved in :',save_dir) 

def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Parse root_dir --save_dir,--arch,--learning_rate --input_layer --hidden_layer --output_layer --epochs,--gpu_on from the command line')
    parser.add_argument('root_dir', default='flowers', type=str, help="gives the data\'s root dir exclude \\train,default flowers")
    parser.add_argument('--save_dir', default='checkpoint.pth', type=str, help="gives the tained modle\'s save path include file name,default checkpoint.pth")
    parser.add_argument('--arch', default='vgg16', choices=['vgg13', 'vgg16', 'resnet18'],type=str, help="gives the ptr-trained model name, default vgg16")
    parser.add_argument('--learning_rate', default=0.003, type=float, help="gives the learning rate,default 0.003")
    parser.add_argument('--input_layer', default=25088, type=int, help="gives the input_layer,default 25088")
    parser.add_argument('--hidden_layer', default=4096, type=int, help="gives the hidden_layer,default 4096")
    parser.add_argument('--output_layer', default=102, type=int, help="gives the output_layer,default 102")
    parser.add_argument('--epochs', default=4, type=int,help="gives the epochs for train,default 4")
    parser.add_argument('--gpu_on', default="Y", help="turn on\(Y\) gpu or not\(N\) , default Y")
    args = parser.parse_args()
    return args


def main():
    # get args from command line
    args = get_command_line_arguments()
    
    train (args.root_dir,args.save_dir,args.arch,args.learning_rate,args.input_layer, args.hidden_layer,args.output_layer,args.epochs,args.gpu_on)

                       
# Call to main function to run the program
if __name__ == "__main__":
    main()

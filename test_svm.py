
curr_dir = '/home/vayzenbe/GitHub_Repos/GiNN'

import sys
sys.path.insert(1, f'{curr_dir}/Models')

from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from ipcl import IPCL0 as IPCL

import os, argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps,  ImageFilter
from itertools import chain
import pandas as pd
import numpy as np
import models
import pdb
import load_without_faces

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])


stim_dir = "/lab_data/behrmannlab/image_sets/imagenet_objects/val"
exclude_im = f"/lab_data/behrmannlab/image_sets/imagenet_face_files.csv"
exclude_folder = f"/lab_data/behrmannlab/image_sets/imagenet_animal_classes.csv"

weights_dir = '/lab_data/behrmannlab/vlad/ginn/model_weights'

def extract_acts(model, im):
    """
    Extracts the activations for a series of images
    """
    model.eval()

    with torch.no_grad():

        im = im.cuda()
        output = model(im)
        output =output.view(output.size(0), -1)
        

    return output

def model_loop(model, loader):
    
    im, label = next(iter(loader))
    
    first_batch = True
    for im, label in loader:
        out = extract_acts(model, im)
        
        if first_batch == True:
            all_out = out
            all_label = label
            first_batch = False
        else:
            all_out = torch.cat((all_out, out), dim=0)
            all_label = torch.cat((all_label, label), dim=0)

    model_acts = all_out.cpu().detach().numpy()
    
    
    return model_acts, all_label.numpy()

model = models.cornet_z()


#the values here are arbitrary just to load the model
model = IPCL(models.__dict__['cornet_z'](out_dim=128), 
                500, # number of imagenet images ORIGINAL is 1281167
                K=4096, 
                T=0.07, 
                out_dim=128, 
                n_samples=5)  

print(model)


checkpoint = torch.load(f'{weights_dir}/cornet_z_cl_15.pth.tar')
model.load_state_dict(checkpoint['state_dict'])


model = model.base_encoder

model = model.cuda()


dataset = load_without_faces.load_stim(stim_dir, exclude_im, exclude_folder, transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False,num_workers = 4, pin_memory=True)

output, label = model_loop(model, loader)

print('starting SVM')
#do SVM
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)

for train_index, test_index in sss.split(output, label):
        
    X_train, X_test = output[train_index], output[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    currScore = clf.score(X_test, y_test)
    print(currScore)
pdb.set_trace()
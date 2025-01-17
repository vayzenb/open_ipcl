import os
import torch
from torch.hub import load_state_dict_from_url
from torchvision.datasets.folder import default_loader
from pathlib import Path 
from torchvision import transforms

from .alexnet_gn import *
from .resnet import *
print('loaded')
url_root = "https://visionlab-pretrainedmodels.s3.amazonaws.com"

"""Vlad nonsense starts here"""
import torch.utils.model_zoo
from .cornet_z import *




"""Vlad nonsesne ends here"""

def build_alexnet_model(weights_url, config):
    
    model = alexnet_gn(out_dim=config['out_dim'], l2norm=config['l2norm'])
    model.config = config
    
    if weights_url is not None:
        print(f"=> loading checkpoint: {Path(weights_url).name}")
        checkpoint = load_state_dict_from_url(weights_url, model_dir=None, map_location=torch.device('cpu'))
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        print("=> state loaded.")
    
    # used for test stimuli (for which we don't want to crop out edges)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    # standard resize and center crop for validation
    model.val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    return model, transform

def cornet_z(out_dim=128):
    return CORnet_Z(out_dim)

def ipcl1():
    model_name = 'ipcl_alpha_alexnet_gn_u128_stack'
    filename = '06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_stack_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 1,
        "type": "ipcl",
        "details": "primary model",
        "aug": "Set 1",
        "top1_knn": 38.4,
        "top1_linear": 39.5,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl2():
    model_name = 'ipcl_alpha_alexnet_gn_u128_rep2'
    filename = '06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_rep2_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 2,
        "type": "ipcl",
        "details": "variation: new code base",
        "aug": "Set 1",
        "top1_knn": 38.4,
        "top1_linear": 39.7,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl3():
    model_name = 'ipcl_alpha_alexnet_gn_u128_redux'
    filename = 'alexnet_gn_dim128_unsupervised_redux_checkpoint_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 3,
        "type": "ipcl",
        "details": "variation: one cycle lr & momentum (73 epochs)",
        "aug": "Set 1",
        "top1_knn": 35.4,
        "top1_linear": 35.7,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl4():
    model_name = 'ipcl_alpha_alexnet_gn_u128_ranger'
    filename = 'alexnet_gn_dim128_unsupervised_ranger_checkpoint_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 4,
        "type": "ipcl",
        "details": "variation: explore ranger (82 epochs)",
        "aug": "Set 1",
        "top1_knn": 37.5,
        "top1_linear": 32.2,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl5():
    model_name = 'ipcl_alpha_alexnet_gn_u128_transforms'
    filename = 'alexnet_gn_dim128_unsupervised_transforms_checkpoint_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 5,
        "type": "ipcl",
        "details": "variation: custom transforms (82 epochs)",
        "aug": "Set 1",
        "top1_knn": 36.9,
        "top1_linear": 38.5,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl6():
    model_name = 'ipcl_alexnet_gn_u128_imagenet'
    filename = 'alexnet_gn_u128_imagenet_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 6,
        "type": "ipcl",
        "details": "ImageNet baseline with new augmentations",
        "aug": "Set 2",
        "top1_knn": 35.1,
        "top1_linear": None,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl7():
    model_name = 'ipcl_alexnet_gn_u128_openimagesv6'
    filename = 'alexnet_gn_u128_openimagesv6_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 7,
        "type": "ipcl",
        "details": "train on independent object dataset, OpenImagesV6",
        "aug": "Set 2",
        "top1_knn": 33.3,
        "top1_linear": None,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl8():
    model_name = 'ipcl_alexnet_gn_u128_places2'
    filename = 'alexnet_gn_u128_places2_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 8,
        "type": "ipcl",
        "details": "train on scene dataset, Places2",
        "aug": "Set 2",
        "top1_knn": 30.9,
        "top1_linear": None,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl9():
    model_name = 'ipcl_alexnet_gn_u128_vggface2'
    filename = 'alexnet_gn_u128_vggface2_lr001_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 9,
        "type": "ipcl",
        "details": "train on face dataset, VggFace2",
        "aug": "Set 2",
        "top1_knn": 12.4,
        "top1_linear": None,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl10():
    model_name = 'ipcl_alexnet_gn_u128_FacesPlacesObjects1281167'
    filename = 'alexnet_gn_u128_FacesPlacesObjects1281167_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 10,
        "type": "ipcl",
        "details": "train on faces-places-objects-1x-ImageNet",
        "aug": "Set 2",
        "top1_knn": 31.6,
        "top1_linear": None,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl11():
    model_name = 'ipcl_alexnet_gn_u128_FacesPlacesObjects1281167x3'
    filename = 'alexnet_gn_u128_FacesPlacesObjects1281167x3_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 11,
        "type": "ipcl",
        "details": "train on faces-places-objects-3x-ImageNet",
        "aug": "Set 2",
        "top1_knn": 33.9,
        "top1_linear": None,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl12():
    model_name = 'ipcl_alpha_alexnet_gn_s1000_imagenet_wus_aug'
    filename = 'alexnet_gn_s1000_imagenet_wus_aug_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 12,
        "type": "category supervised",
        "details": "trained with 5 augmentations per image to match IPCL",
        "aug": "Set 1",
        "top1_knn": 58.8,
        "top1_linear": 55.7,
        "out_dim": 1000,
        "l2norm": False,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl13():
    model_name = 'wusnet_alexnet_gn_s1000'
    filename = 'alexnet_gn_supervised_final.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "wusnet", filename)
    
    config = {
        "ref#": 13,
        "type": "category supervised",
        "details": "trained with single augmentation per image",
        "aug": "Set 1",
        "top1_knn": 55.5,
        "top1_linear": 54.5,
        "out_dim": 1000,
        "l2norm": False,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl14():
    model_name = 'ipcl_alexnet_gn_s1000_imagenet'
    filename = 'alexnet_gn_s1000_imagenet_base_aug3_blur0_rot0_w0_d0_std02_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)
    
    config = {
        "ref#": 14,
        "type": "category supervised",
        "details": "ImageNet baseline with new augmentations",
        "aug": "Set 2",
        "top1_knn": 56.0,
        "top1_linear": None,
        "out_dim": 1000,
        "l2norm": False,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl15():
    model_name = 'ipcl_alexnet_gn_s1000_imagenet_rep1'
    filename = 'alexnet_gn_s1000_imagenet_base_aug3_blur0_rot0_w0_d0_std02_rep1_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, "project_instancenet", "ipcl", filename)    
    
    config = {
        "ref#": 15,
        "type": "category supervised",
        "details": "primary model",
        "aug": "Set 2",
        "top1_knn": 56.0,
        "top1_linear": None,
        "out_dim": 1000,
        "l2norm": False,
        "mean": [0.5, 0.5, 0.5], 
        "std": [0.2, 0.2, 0.2]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

def ipcl16():
    model_name = 'ipcl_alpha_alexnet_gn_u128_random'
    filename = ''
    weights_url = None    
    
    config = {
        "ref#": 16,
        "type": "untrained",
        "details": "untrained model with random weights and biases",
        "aug": "-",
        "top1_knn": 3.5,
        "top1_linear": 7.2,
        "out_dim": 128,
        "l2norm": True,
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    }
    print(config)
    
    model, transform = build_alexnet_model(weights_url, config)
    
    return model, transform

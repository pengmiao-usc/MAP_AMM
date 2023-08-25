import numpy as np
import json
import os
import yaml

from models.mlp_mixer import MLPMixer
from models.mlp_simple import MLP
from models.mlp_teacher import MLPTeacher
from models.r import resnet34 

def select_model(option):
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
        
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    
    if option == "mm":
        channels = params["model"][f"{option}"]["channels"]
        dim = params["model"][f"{option}"]["dim"]
        depth = params["model"][f"{option}"]["depth"]
        return MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size[1],
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
    elif option == "ms":
        input_size = params["model"][f"{option}"]["input_size"]
        hidden_size = params["model"][f"{option}"]["hidden_size"]
        num_classes = params["model"][f"{option}"]["num_classes"]
        return MLP(
            input_size = input_size,
            hidden_size = hidden_size,
            num_classes = num_classes
        )
    elif option == "mt":
        input_size = params["model"][f"{option}"]["input_size"]
        hidden_size = params["model"][f"{option}"]["hidden_size"]
        num_classes = params["model"][f"{option}"]["num_classes"]
        return MLPTeacher(
            input_size = input_size,
            hidden_size = hidden_size,
            num_classes = num_classes
        )
    elif option == "rs":
        channels = params["model"][option]["channels"]
        dim = params["model"][option]["dim"]
        return resnet34(num_classes, channels) 
    elif option == "rt":
        channels = params["model"][option]["channels"]
        dim = params["model"][option]["dim"]
        return resnet34(num_classes, channels) 


def replace_directory(path, new_directory):
    parts = path.split('/')
    parts[-2] = new_directory
    new_path = '/'.join(parts)
    return new_path



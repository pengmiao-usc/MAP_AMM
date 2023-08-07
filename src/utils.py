import numpy as np
import yaml
from models.mlp_mixer import MLPMixer
from models.mlp_simple import MLP

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



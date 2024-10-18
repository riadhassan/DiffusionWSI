import torch.nn as nn
from networks.demo_net import ImageTranslationCNN

def model_wrapper(model_name):
    # in_channel = 1
    # if (conf.dataset == "LCTSC"):
    #     out_channel = 6
    # elif (conf.dataset == "SegThor"):
    #     out_channel = 5

    architecture = {
        "ImageTranslationCNN": ImageTranslationCNN(),
    }

    model = architecture[model_name]

    return model

def loss_wrapper(loss_name):

    loss_function  = {
        "MSE": nn.MSELoss(),
    }

    criterion = loss_function[loss_name]

    return criterion
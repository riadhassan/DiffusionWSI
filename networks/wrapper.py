import torch.nn as nn
from networks.demo_net import ImageTranslationCNN
from networks.koopman_network import KoopmanUNet
from networks.Unet import UNet
def model_wrapper(model_name):
    # in_channel = 1
    # if (conf.dataset == "LCTSC"):
    #     out_channel = 6
    # elif (conf.dataset == "SegThor"):
    #     out_channel = 5

    architecture = {
        "ImageTranslationCNN": ImageTranslationCNN(),
        "KoopmanUNet": KoopmanUNet(n_channels=3, n_classes=3),
        "UNet": UNet(n_channels=3, n_classes=3),
    }

    model = architecture[model_name]

    print(f"Load {model_name} model")

    return model

def loss_wrapper(loss_name):

    loss_function  = {
        "MSE": nn.MSELoss(),
    }
    try:
        criterion = loss_function[loss_name]
    except:
        criterion = nn.MSELoss()

    return criterion
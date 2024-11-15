import torch.nn as nn
from networks.demo_net import ImageTranslationCNN
from networks.koopman_network import UNet_koopman
from networks.Unet import UNet
def model_wrapper(model_name):
    # in_channel = 1
    # if (conf.dataset == "LCTSC"):
    #     out_channel = 6
    # elif (conf.dataset == "SegThor"):
    #     out_channel = 5

    architecture = {
        "ImageTranslationCNN": ImageTranslationCNN(),
        "KoopmanUNet": UNet_koopman(in_channels=3, out_channels=3),
        "UNet": UNet(in_channels=3, out_channels=3),
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
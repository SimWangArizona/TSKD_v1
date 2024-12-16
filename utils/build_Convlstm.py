import torch
from torch.autograd import Variable
from utils.config_loader import load_config
import os
from conv_lstm_model import net_params_wrn401,net_params_shufflev1_change,net_params_resnet,net_params_vgg
from conv_lstm_model.encoder import Encoder
from conv_lstm_model.forecaster import Forecaster
from conv_lstm_model.model import EF
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from .build_utils import build_lstm_review
from ts_model.wide_resnet_cifar import wrn_40_1
from .build_utils import at

file_path = os.path.dirname(__file__)
default_config = os.path.join(file_path, "config.yaml")
config = load_config(file_path, default_config_path=default_config)
lstm_config = config['LSTM']

def build_convlstm(model):
    # print(type(model))
    if 'wrn' in model:
        # print('wrnnet')
        encoder1 = Encoder(net_params_wrn401.convlstm_encoder1_params[0], net_params_wrn401.convlstm_encoder1_params[1]).to(config['device'])
        # print(encoder)
        forecaster1 = Forecaster(net_params_wrn401.convlstm_forecaster1_params[0], net_params_wrn401.convlstm_forecaster1_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster1 = EF(encoder1, forecaster1).to(config['device'])

        encoder2 = Encoder(net_params_wrn401.convlstm_encoder2_params[0], net_params_wrn401.convlstm_encoder2_params[1]).to(config['device'])
        # print(encoder)
        forecaster2 = Forecaster(net_params_wrn401.convlstm_forecaster2_params[0], net_params_wrn401.convlstm_forecaster2_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster2 = EF(encoder2, forecaster2).to(config['device'])

        encoder3 = Encoder(net_params_wrn401.convlstm_encoder3_params[0], net_params_wrn401.convlstm_encoder3_params[1]).to(config['device'])
        # print(encoder)
        forecaster3 = Forecaster(net_params_wrn401.convlstm_forecaster3_params[0], net_params_wrn401.convlstm_forecaster3_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster3 = EF(encoder3, forecaster3).to(config['device'])

        ConvLSTMs = [encoder_forecaster1, encoder_forecaster2, encoder_forecaster3]

        return ConvLSTMs

    elif 'shuffle' in model:
        # print('shufflev1net')
        encoder1 = Encoder(net_params_shufflev1_change.convlstm_encoder1_params[0],
                           net_params_shufflev1_change.convlstm_encoder1_params[1]).to(config['device'])
        # print(encoder)
        forecaster1 = Forecaster(net_params_shufflev1_change.convlstm_forecaster1_params[0],
                                 net_params_shufflev1_change.convlstm_forecaster1_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster1 = EF(encoder1, forecaster1).to(config['device'])

        encoder2 = Encoder(net_params_shufflev1_change.convlstm_encoder2_params[0],
                           net_params_shufflev1_change.convlstm_encoder2_params[1]).to(config['device'])
        # print(encoder)
        forecaster2 = Forecaster(net_params_shufflev1_change.convlstm_forecaster2_params[0],
                                 net_params_shufflev1_change.convlstm_forecaster2_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster2 = EF(encoder2, forecaster2).to(config['device'])

        encoder3 = Encoder(net_params_shufflev1_change.convlstm_encoder3_params[0],
                           net_params_shufflev1_change.convlstm_encoder3_params[1]).to(config['device'])
        # print(encoder)
        forecaster3 = Forecaster(net_params_shufflev1_change.convlstm_forecaster3_params[0],
                                 net_params_shufflev1_change.convlstm_forecaster3_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster3 = EF(encoder3, forecaster3).to(config['device'])

        ConvLSTMs = [encoder_forecaster1, encoder_forecaster2, encoder_forecaster3]
        return ConvLSTMs

    elif 'resnet' in model:
        encoder1 = Encoder(net_params_resnet.convlstm_encoder1_params[0],
                           net_params_resnet.convlstm_encoder1_params[1]).to(config['device'])
        # print(encoder)
        forecaster1 = Forecaster(net_params_resnet.convlstm_forecaster1_params[0],
                                 net_params_resnet.convlstm_forecaster1_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster1 = EF(encoder1, forecaster1).to(config['device'])

        encoder2 = Encoder(net_params_resnet.convlstm_encoder2_params[0],
                           net_params_resnet.convlstm_encoder2_params[1]).to(config['device'])
        # print(encoder)
        forecaster2 = Forecaster(net_params_resnet.convlstm_forecaster2_params[0],
                                 net_params_resnet.convlstm_forecaster2_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster2 = EF(encoder2, forecaster2).to(config['device'])

        encoder3 = Encoder(net_params_resnet.convlstm_encoder3_params[0],
                           net_params_resnet.convlstm_encoder3_params[1]).to(config['device'])
        # print(encoder)
        forecaster3 = Forecaster(net_params_resnet.convlstm_forecaster3_params[0],
                                 net_params_resnet.convlstm_forecaster3_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster3 = EF(encoder3, forecaster3).to(config['device'])

        ConvLSTMs = [encoder_forecaster1, encoder_forecaster2, encoder_forecaster3]
        return ConvLSTMs
    elif "vgg" in model:
        encoder1 = Encoder(net_params_vgg.convlstm_encoder1_params[0],
                           net_params_vgg.convlstm_encoder1_params[1]).to(config['device'])
        # print(encoder)
        forecaster1 = Forecaster(net_params_vgg.convlstm_forecaster1_params[0],
                                 net_params_vgg.convlstm_forecaster1_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster1 = EF(encoder1, forecaster1).to(config['device'])

        encoder2 = Encoder(net_params_vgg.convlstm_encoder2_params[0],
                           net_params_vgg.convlstm_encoder2_params[1]).to(config['device'])
        # print(encoder)
        forecaster2 = Forecaster(net_params_vgg.convlstm_forecaster2_params[0],
                                 net_params_vgg.convlstm_forecaster2_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster2 = EF(encoder2, forecaster2).to(config['device'])

        encoder3 = Encoder(net_params_vgg.convlstm_encoder3_params[0],
                           net_params_vgg.convlstm_encoder3_params[1]).to(config['device'])
        # print(encoder)
        forecaster3 = Forecaster(net_params_vgg.convlstm_forecaster3_params[0],
                                 net_params_vgg.convlstm_forecaster3_params[1]).to(config['device'])
        # print(forecaster)
        encoder_forecaster3 = EF(encoder3, forecaster3).to(config['device'])

        ConvLSTMs = [encoder_forecaster1, encoder_forecaster2, encoder_forecaster3]

        return ConvLSTMs



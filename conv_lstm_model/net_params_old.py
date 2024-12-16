import sys
import torch
from .forecaster import Forecaster
from .encoder import Encoder
from utils.config_loader import load_config

from collections import OrderedDict
import os
import numpy as np
from .convLSTM import ConvLSTM


file_path = os.path.dirname(os.path.dirname(__file__))
default_config = os.path.join(file_path, "config.yaml")
config = load_config(file_path, default_config_path=default_config)

lstm_config = config['LSTM']
encoder_config = lstm_config['encoder']
forecaster_config = lstm_config['forecaster']

batch_size = config['dataloader']['batch_size']
IN_LEN = lstm_config['in_seq_len']
OUT_LEN = lstm_config['out_seq_len']

# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=encoder_config['ConvLSTM_layer1']['input_channel'],
                 num_filter=encoder_config['ConvLSTM_layer1']['num_filter'], b_h_w=(
            batch_size, encoder_config['ConvLSTM_layer1']['height'], encoder_config['ConvLSTM_layer1']['width']),
                 kernel_size=encoder_config['ConvLSTM_layer1']['kernel_size'],
                 stride=encoder_config['ConvLSTM_layer1']['stride'],
                 padding=encoder_config['ConvLSTM_layer1']['padding']),
        ConvLSTM(input_channel=encoder_config['ConvLSTM_layer2']['input_channel'],
                 num_filter=encoder_config['ConvLSTM_layer2']['num_filter'], b_h_w=(
            batch_size, encoder_config['ConvLSTM_layer2']['height'], encoder_config['ConvLSTM_layer2']['width']),
                 kernel_size=encoder_config['ConvLSTM_layer2']['kernel_size'],
                 stride=encoder_config['ConvLSTM_layer2']['stride'],
                 padding=encoder_config['ConvLSTM_layer2']['padding']),
        ConvLSTM(input_channel=encoder_config['ConvLSTM_layer3']['input_channel'],
                 num_filter=encoder_config['ConvLSTM_layer3']['num_filter'], b_h_w=(
            batch_size, encoder_config['ConvLSTM_layer3']['height'], encoder_config['ConvLSTM_layer3']['width']),
                 kernel_size=encoder_config['ConvLSTM_layer3']['kernel_size'],
                 stride=encoder_config['ConvLSTM_layer3']['stride'],
                 padding=encoder_config['ConvLSTM_layer3']['padding']),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=forecaster_config['ConvLSTM_layer1']['input_channel'],
                 num_filter=forecaster_config['ConvLSTM_layer1']['num_filter'], b_h_w=(
            batch_size, forecaster_config['ConvLSTM_layer1']['height'], forecaster_config['ConvLSTM_layer1']['width']),
                 kernel_size=forecaster_config['ConvLSTM_layer1']['kernel_size'],
                 stride=forecaster_config['ConvLSTM_layer1']['stride'],
                 padding=forecaster_config['ConvLSTM_layer1']['padding']),
        ConvLSTM(input_channel=forecaster_config['ConvLSTM_layer2']['input_channel'],
                 num_filter=forecaster_config['ConvLSTM_layer2']['num_filter'], b_h_w=(
            batch_size, forecaster_config['ConvLSTM_layer2']['height'], forecaster_config['ConvLSTM_layer2']['width']),
                 kernel_size=forecaster_config['ConvLSTM_layer2']['kernel_size'],
                 stride=forecaster_config['ConvLSTM_layer2']['stride'],
                 padding=forecaster_config['ConvLSTM_layer2']['padding']),
        ConvLSTM(input_channel=forecaster_config['ConvLSTM_layer3']['input_channel'],
                 num_filter=forecaster_config['ConvLSTM_layer3']['num_filter'], b_h_w=(
            batch_size, forecaster_config['ConvLSTM_layer3']['height'], forecaster_config['ConvLSTM_layer3']['width']),
                 kernel_size=forecaster_config['ConvLSTM_layer3']['kernel_size'],
                 stride=forecaster_config['ConvLSTM_layer3']['stride'],
                 padding=forecaster_config['ConvLSTM_layer3']['padding']),
    ]
]

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

# build model
convlstm_encoder1_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]})
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster1_params = [
    [
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 4, 2, 1],  #in_chaanels, out_channels, kernel_size, stride, padding
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
    ]
]



convlstm_encoder2_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]})
    ],

    [
        ConvLSTM(input_channel=8, num_filter=32, b_h_w=(batch_size, 8, 8),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster2_params = [
    [
        OrderedDict({
            'deconv3_leaky_1': [32, 8, 4, 2, 1],  #in_chaanels, out_channels, kernel_size, stride, padding
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=32, num_filter=32, b_h_w=(batch_size, 8, 8),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_encoder3_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]})
    ],

    [
        ConvLSTM(input_channel=8, num_filter=32, b_h_w=(batch_size, 4, 4),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster3_params = [
    [
        OrderedDict({
            'deconv3_leaky_1': [32, 8, 4, 2, 1],  #in_chaanels, out_channels, kernel_size, stride, padding
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=32, num_filter=32, b_h_w=(batch_size, 4, 4),
                 kernel_size=3, stride=1, padding=1),
    ]
]

# convlstm_encoder4_params = [
#     [
#         OrderedDict({'conv1_leaky_1': [1, 8, 3, 2, 1]})
#     ],
#
#     [
#         ConvLSTM(input_channel=8, num_filter=32, b_h_w=(batch_size, 4, 4),
#                  kernel_size=3, stride=1, padding=1),
#     ]
# ]
#
# convlstm_forecaster4_params = [
#     [
#         OrderedDict({
#             'deconv3_leaky_1': [32, 8, 4, 2, 1],  #in_chaanels, out_channels, kernel_size, stride, padding
#             'conv3_leaky_2': [8, 8, 3, 1, 1],
#             'conv3_3': [8, 1, 1, 1, 0]
#         }),
#     ],
#
#     [
#         ConvLSTM(input_channel=32, num_filter=32, b_h_w=(batch_size, 4, 4),
#                  kernel_size=3, stride=1, padding=1),
#     ]
# ]
import sys
from torch import nn
import torch
from utils.makelayer_utils import make_layers
sys.path.append('..')
from utils.config_loader import load_config
import logging
import os

file_path = os.path.dirname(os.path.dirname(__file__))
default_config = os.path.join(file_path, "config.yaml")
config = load_config(file_path, default_config_path=default_config)
lstm_config = config['LSTM']

class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        # print(self.blocks)
        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=lstm_config['out_seq_len'])
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        # print(input.size())
        input = subnet(input)
        # print(input.size())
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage1'),
                                      getattr(self, 'rnn1'))
        # for i in list(range(1, self.blocks))[::-1]:
        #     input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
        #                                            getattr(self, 'rnn' + str(i)))
        # print(input)
        return input

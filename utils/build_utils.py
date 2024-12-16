import torch
import torch.nn as nn
import torch.nn.functional as F
from ts_model.wide_resnet_cifar import wrn
from ts_model.shufflenetv1 import ShuffleV1
from ts_model.shufflenetv2 import ShuffleV2
from ts_model.resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from ts_model.vgg import vgg13_bn,vgg8_bn
#from .vgg import build_vgg_backbone
from ts_model.resnet import ResNet50
import os
from utils.config_loader import load_config

file_path = os.path.dirname(__file__)
default_config = os.path.join(file_path, "config.yaml")
config = load_config(file_path, default_config_path=default_config)
lstm_config = config['LSTM']

# attention map
def at(x):
    return F.normalize(x.pow(2).mean(1)).view(x.size(0),1,x.size(2),x.size(3))


# attention loss
def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


# knowledge increment
def knowledge_increment(x,y):
    return torch.abs(x - y)

# previous_features = [[0 for col in range(num_branch)] for row in range(in_seq_len)]  # 创建一个isl*nb的二维列表，每个初值都是0

def build_increment_sequence(previous_features,current_features):
    in_seq_len = len(previous_features)
    num_branch = len(previous_features[0])
    increment_sequence = [[0 for col in range(num_branch)] for row in range(in_seq_len)]
    for i in range(num_branch):
        current_branch = current_features[i]
        for j in range(in_seq_len):
            if j != in_seq_len - 1:
                increment = knowledge_increment(previous_features[j][i],previous_features[j+1][i])
            else:
                increment = knowledge_increment(current_branch,previous_features[j][i])
            increment_sequence[j][i] = increment

    return increment_sequence

# KD loss
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


class lstm_review(nn.Module):
    def __init__(self, student, ConvLSTMs):
        super(lstm_review, self).__init__()
        self.student = student

        convlstm_list = nn.ModuleList()
        for idx,convlstm in enumerate(ConvLSTMs):
            convlstm_list.append(convlstm)

        self.ConvLSTMs = convlstm_list
        self.in_seq_len = lstm_config['in_seq_len']
        self.num_branch = lstm_config['num_branch']
    def forward(self, x, previous_features = None):
        student_output = self.student(x, is_feat=True)
        # logit = student_output[1]
        current_features = student_output[0]
        current_features = current_features[0:self.num_branch]
        '''Current features to attention maps'''
        for idx,item in enumerate(current_features):
            current_features[idx] = at(item)

        if previous_features != None:
            increment_seq = build_increment_sequence(previous_features,current_features)
            # print(len(increment_seq))
            seqs = []
            seq1 = torch.stack([increment_seq[0][0],increment_seq[1][0],increment_seq[2][0]]
                               ,dim=0)
            seqs.append(seq1)
            seq2 = torch.stack([increment_seq[0][1],increment_seq[1][1],increment_seq[2][1]]
                               ,dim=0)
            seqs.append(seq2)
            seq3 = torch.stack([increment_seq[0][2],increment_seq[1][2],increment_seq[2][2]]
                               ,dim=0)
            seqs.append(seq3)


            ConvLSTM_output = []
            # assert len(increment_seq) == len(self.ConvLSTMs)
            for seq,ConvLSTM in zip(seqs,self.ConvLSTMs):
                # print('seq shape:',seq.shape)
                # print(ConvLSTM)
                ConvLSTM_output.append(ConvLSTM(seq))
            return student_output, ConvLSTM_output
        else:
            return student_output


def build_lstm_review(model,num_classes,ConvLSTMs,teacher='wrn-40-2'):
    out_shapes = None
    if 'x4' in model:
        student = build_resnetx4_backbone(depth=int(model[6:-2]), num_classes=num_classes)
    elif 'resnet' in model:
        student = build_resnet_backbone(depth=int(model[6:]), num_classes=num_classes)
    elif 'vgg' in model:
        student = vgg8_bn( num_classes=num_classes).cuda()
    elif 'wrn' in model:
        student = wrn(depth=int(model[4:6]), widen_factor=int(model[-1:]), num_classes=num_classes)
    else:
        assert False
    backbone = lstm_review(
        student=student,
        ConvLSTMs=ConvLSTMs
    )
    return backbone
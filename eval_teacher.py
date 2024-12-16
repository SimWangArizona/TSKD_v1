import torch
from torch.autograd import Variable
from utils.config_loader import load_config
import os
from conv_lstm_model.net_params_wrn401 import convlstm_encoder1_params,convlstm_forecaster1_params,convlstm_encoder3_params,convlstm_forecaster3_params
from conv_lstm_model.encoder import Encoder
from conv_lstm_model.forecaster import Forecaster
from conv_lstm_model.model import EF
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from ts_model.resnet_cifar import resnet110
from ts_model.resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from ts_model.wide_resnet_cifar import wrn
from ts_model.wide_resnet_cifar import wrn
from ts_model.vgg import vgg13_bn
import argparse
from ts_model.resnet_cifar import resnet8x4
# model = 'wrn-40-2'
# cnn = wrn(depth=int(model[4:6]), widen_factor=int(model[-1:]), num_classes=100)
# cnn = vgg13_bn(num_classes = 100)
# # print(cnn)
# cnn.load_state_dict(torch.load('teacher_ckpt/cifar100_vgg13__baseline1_best.pt')["model"])
# cnn = cnn.cuda()
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    )
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=240,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--suffix', type=str, default='lstm_review_kd',
                    help='label')
parser.add_argument('--teacher', type=str, default='resnet56',
                    help='teacher ts_model')

args = parser.parse_args()

if 'x4' in args.teacher:
    teacher = build_resnetx4_backbone(depth = int(args.teacher[6:-2]), num_classes=100)
elif 'resnet' in args.teacher:
    teacher = build_resnet_backbone(depth = int(args.teacher[6:]), num_classes=100)
elif 'wrn' in args.teacher:
    teacher = wrn(depth = int(args.teacher[4:6]), widen_factor = int(args.teacher[-1:]), num_classes=100)
elif 'vgg' in args.teacher:
    teacher = vgg13_bn(num_classes = 100).cuda()

resume = args.dataset + '_' + args.teacher + '__' + 'baseline1' + '_best.pt'
teacher.load_state_dict(torch.load("teacher_ckpt/"+resume))

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

test_dataset = datasets.CIFAR100(root='CIFAR-100/',train=False,transform=test_transform,download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=128,
                                          shuffle=False,pin_memory=True,num_workers=1)


def test(loader):
    teacher.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            pred = teacher(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    teacher.train()
    return val_acc

if __name__ == '__main__':
    print('Teacher model is:', args.teacher)
    print('Accuracy on CIAFR-100:',test(test_loader)*100)
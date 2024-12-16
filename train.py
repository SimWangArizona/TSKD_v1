from utils.config_loader import load_config
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
from torchvision import datasets, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
file_path = os.path.dirname(__file__)
default_config = os.path.join(file_path, "config.yaml")
config = load_config(file_path, default_config_path=default_config)
lstm_config = config['LSTM']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    )
parser.add_argument('--model', '-a', default='resnet56',
                    )
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=240,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay ratio')
parser.add_argument('--lr_adjust_step', default=[150, 180, 210], type=int, nargs='+',
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--suffix', type=str, default='lstm_review_kd',
                    help='label')
parser.add_argument('--test', action='store_true', default=False,
                    help='test')
parser.add_argument('--resume', type=str, default='',
                    help='resume')

parser.add_argument('--teacher', type=str, default='wrn-40-2',
                    help='teacher ts_model')
parser.add_argument('--teacher-weight', type=str, default='teacher_ckpt/cifar100_wrn-40-2__baseline1_best.pt',
                    help='teacher ts_model weight path')
parser.add_argument('--kd-loss-weight', type=float, default=5.0,
                    help='review kd loss weight')
parser.add_argument('--kd-warm-up', type=float, default=20.0,
                    help='feature konwledge distillation loss weight warm up epochs')

parser.add_argument('--use_kl', action='store_true', default=True,
                    help='use kl kd loss')
parser.add_argument('--kl-loss-weight', type=float, default=1.0,
                    help='kl konwledge distillation loss weight')
parser.add_argument('-T', type=float, default=4.0,
                    help='knowledge distillation loss temperature')
parser.add_argument('--ce-loss-weight', type=float, default=1.0,
                    help='cross entropy loss weight')

args = parser.parse_args()
assert torch.cuda.is_available()

cudnn.deterministic = True
cudnn.benchmark = False
if args.seed == 0:
    args.seed = np.random.randint(1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)


from utils.misc import *
from utils import build_utils
from ts_model.resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from ts_model.wide_resnet_cifar import wrn
from ts_model.vgg import vgg13_bn,vgg8_bn

test_id = args.dataset + '_' + args.model + '_' + args.teacher + '_' + args.suffix
filename = 'logs/' + test_id + '.txt'
logger = Logger(args=args, filename=filename)
print(args)

# Image Preprocessing
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

# dataset
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='CIFAR-10/',train=True,transform=train_transform,download=True)
    test_dataset = datasets.CIFAR10(root='CIFAR-10/',train=False,transform=test_transform,download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='CIFAR-100/',train=True,transform=train_transform,download=True)
    test_dataset = datasets.CIFAR100(root='CIFAR-100/',train=False,transform=test_transform,download=True)
else:
    assert False
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,
                                          shuffle=False,pin_memory=False,num_workers=1)

# ConvLSTM
from utils.build_Convlstm import build_convlstm
ConvLSTMs = build_convlstm(args.model)

# divide the training into M,G,R sets
points = []
for i in range(240):
    if i%5 == 0:
        points.append(i)
first_order_mempoints = points[::4]
second_order_mempoints = points[1::4]
third_order_mempoints = points[2::4]
review_points = points[3::4]
memory_points = sorted(first_order_mempoints + second_order_mempoints + third_order_mempoints)


# teacher model
if 'x4' in args.teacher:
    teacher = build_resnetx4_backbone(depth = int(args.teacher[6:-2]), num_classes=num_classes)
elif 'resnet' in args.teacher:
    teacher = build_resnet_backbone(depth = int(args.teacher[6:]), num_classes=num_classes)
elif 'wrn' in args.teacher:
    teacher = wrn(depth = int(args.teacher[4:6]), widen_factor = int(args.teacher[-1:]), num_classes=num_classes)
elif 'vgg' in args.teacher:
    teacher = vgg13_bn(num_classes = num_classes).cuda()
elif args.teacher == '':
    teacher = None
else:
    assert False
if teacher is not None:
    load_teacher_weight(teacher, args.teacher_weight, args.teacher)

# model
if teacher is not None:
    cnn = build_utils.build_lstm_review(model=args.model,ConvLSTMs=ConvLSTMs,num_classes=num_classes)

else:
    assert False

if 'shuffle' in args.model or 'mobile' in args.model:
    args.lr = 0.02

trainable_parameters = nn.ModuleList()
trainable_parameters.append(cnn)
MSE_loss = nn.MSELoss().cuda()
criterion = nn.CrossEntropyLoss().cuda()
kl_criterion = build_utils.DistillKL(args.T)
wd = args.wd
lr = args.lr
cnn_optimizer = torch.optim.SGD(trainable_parameters.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=wd)

def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
        if teacher is not None:
            fs, pred = pred

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

if args.test:
    cnn.load_state_dict(torch.load(args.resume))
    print(test(test_loader))
    exit()

if __name__ == '__main__':
    # train
    best_acc = 0.0
    st_time = time.time()
    for epoch in range(args.epochs):
        loss_avg = {}
        correct = 0.
        total = 0.
        cnt_ft = {}
        if epoch in memory_points:
            point_state = 'memory_points'
            # freeze convlstm
            for name, param in cnn.named_parameters():
                if 'ConvLSTMs' in name:
                    param.requires_grad = False

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()
                cnn.zero_grad()
                losses = {}
                # print(cnn)
                student_output = cnn(images)
                pred = student_output[1]
                t_features, t_pred = teacher(images, is_feat = True, preact=True)
                t_features = t_features[1:]
                if args.use_kl:
                    losses['kl_kd_loss'] = kl_criterion(pred, t_pred) * args.kl_loss_weight

                # print(pred.shape)
                xentropy_loss = criterion(pred, labels)

                losses['cls_loss'] = xentropy_loss * args.ce_loss_weight
                loss = sum(losses.values())
                loss.backward()
                cnn_optimizer.step()

                for key in losses:
                    if not key in loss_avg:
                        loss_avg[key] = AverageMeter()
                    else:
                        loss_avg[key].update(losses[key])

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

            # save memory points
            if epoch in first_order_mempoints:
                torch.save(cnn,'memory_points/' + args.model + '_first_order_mem.pt')
            elif epoch in second_order_mempoints:
                torch.save(cnn,'memory_points/' + args.model + '_second_order_mem.pt')
            elif epoch in third_order_mempoints:
                torch.save(cnn, 'memory_points/' + args.model + '_third_order_mem.pt')

        elif epoch in review_points:
            point_state = 'review_points'
            # unfreeze convlstm
            for name, param in cnn.named_parameters():
                if 'ConvLSTMs' in name:
                    param.requires_grad = True
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()

                # load memory points
                first_order_mem = torch.load('memory_points/' + args.model + '_first_order_mem.pt')
                first_order_mem.to(config['device'])
                second_order_mem = torch.load('memory_points/' + args.model + '_second_order_mem.pt')
                second_order_mem.to(config['device'])
                third_order_mem = torch.load('memory_points/' + args.model + '_third_order_mem.pt')
                third_order_mem.to(config['device'])

                # stack features
                with torch.no_grad():
                    first_order_feat,_ = first_order_mem(images)
                    second_order_feat,_ = second_order_mem(images)
                    third_order_feat,_ = third_order_mem(images)

                previous_features = [[0 for col in range(lstm_config['num_branch'])] for row in
                                     range(lstm_config['in_seq_len'])]

                previous_features[0][0] = first_order_feat[0]
                previous_features[0][1] = first_order_feat[1]
                previous_features[0][2] = first_order_feat[2]
                previous_features[1][0] = second_order_feat[0]
                previous_features[1][1] = second_order_feat[1]
                previous_features[1][2] = second_order_feat[2]
                previous_features[2][0] = third_order_feat[0]
                previous_features[2][1] = third_order_feat[1]
                previous_features[2][2] = third_order_feat[2]

                for i in range(lstm_config['in_seq_len']):
                    for j in range(lstm_config['num_branch']):
                        previous_features[i][j] = build_utils.at(previous_features[i][j])

                cnn.zero_grad()
                losses = {}

                student_output, convlstm_output = cnn(images,previous_features)
                s_features = student_output[0]
                s_features = s_features[0:lstm_config['num_branch']]

                pred = student_output[1]

                t_features, t_pred = teacher(images, is_feat=True, preact=True)
                t_features = t_features[0:lstm_config['num_branch']]
                # if the spatail dimension does not match, process avg pool.
                for idx,(s,t) in enumerate(zip(s_features,t_features)):
                    if s.shape[2] != t.shape[2]:
                        t_features[idx] = F.adaptive_avg_pool2d(t,(s.shape[2],s.shape[3]))
                    else:
                        continue

                ts_increment = []
                for s,t in zip(s_features,t_features):
                    ts_increment.append(build_utils.knowledge_increment(build_utils.at(s),build_utils.at(t)))

                lstm_review_loss = 0
                for pred_inc,ts_inc in zip(convlstm_output,ts_increment):
                    # print('pred inc shape',pred_inc.shape)
                    # print('ts inc shape',ts_inc.shape)
                    pred_inc = torch.reshape(pred_inc,ts_inc.shape)
                    lstm_review_loss+= MSE_loss(pred_inc,ts_inc)

                losses['lstm_review_loss'] = lstm_review_loss

                if args.use_kl:
                    losses['kl_kd_loss'] = kl_criterion(pred, t_pred) * args.kl_loss_weight

                xentropy_loss = criterion(pred, labels)

                losses['cls_loss'] = xentropy_loss * args.ce_loss_weight
                loss = sum(losses.values())
                loss.backward()
                cnn_optimizer.step()

                for key in losses:
                    if not key in loss_avg:
                        loss_avg[key] = AverageMeter()
                    else:
                        loss_avg[key].update(losses[key])

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

        # not memory points or review points
        else:
            point_state = 'general_points'
            for name, param in cnn.named_parameters():
                if 'ConvLSTMs' in name:
                    param.requires_grad = False

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()
                cnn.zero_grad()
                losses = {}

                student_output= cnn(images)
                pred = student_output[1]
                t_features, t_pred = teacher(images, is_feat = True, preact=True)
                t_features = t_features[1:]
                if args.use_kl:
                    losses['kl_kd_loss'] = kl_criterion(pred, t_pred) * args.kl_loss_weight

                xentropy_loss = criterion(pred, labels)

                losses['cls_loss'] = xentropy_loss * args.ce_loss_weight
                loss = sum(losses.values())
                loss.backward()
                cnn_optimizer.step()

                for key in losses:
                    if not key in loss_avg:
                        loss_avg[key] = AverageMeter()
                    else:
                        loss_avg[key].update(losses[key])

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

        test_acc = test(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '_best.pt')
        lr = lr_schedule(lr, epoch, cnn_optimizer, args)

        loss_avg = {k: loss_avg[k].val for k in loss_avg}
        row = { 'epoch': str(epoch),
                'point_state': point_state,
                'train_acc': '%.2f'%(accuracy*100),
                'test_acc': '%.2f'%(test_acc*100),
                'best_acc': '%.2f'%(best_acc*100),
                'lr': '%.5f'%(lr),
                'loss': '%.5f'%(sum(loss_avg.values())),
                }
        loss_avg = {k: '%.5f'%loss_avg[k] for k in loss_avg}
        row.update(loss_avg)
        row.update({
                'time': format_time(time.time()-st_time),
                'eta': format_time((time.time()-st_time)/(epoch+1)*(args.epochs-epoch-1)),
                })
        print(row)
        logger.writerow(row)

    torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
    logger.close()

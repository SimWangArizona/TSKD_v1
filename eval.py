import argparse
from torchvision import datasets, transforms
from utils.misc import *
from utils import build_utils
from utils.build_Convlstm import build_convlstm

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    )
parser.add_argument('--model', '-a', default='resnet20',
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

ConvLSTMs = build_convlstm(model=args.model)
# print(ConvLSTMs)
cnn = build_utils.build_lstm_review(model=args.model,ConvLSTMs=ConvLSTMs,num_classes=100)
resume = args.dataset + '_' + args.model + '_' + args.teacher + '_' + args.suffix + '_best.pt'
cnn.load_state_dict(torch.load('checkpoints/'+resume))

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
                                           shuffle=True, pin_memory=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,
                                          shuffle=False,pin_memory=True,num_workers=1)

def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            student_output = cnn(images)
            pred = student_output[1]

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    # print(correct)
    val_acc = correct / total
    cnn.train()
    return val_acc

if __name__ == '__main__':
    print('Teacher model is:',args.teacher)
    print('Student is:',args.model)
    print('Accuracy on CIAFR-100:',test(test_loader)*100)
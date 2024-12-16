This repo is an official implementation of "Temporal Supervised Knowledge Distillation".

### Installation

Environments:

- Python 3.9
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

### Getting started


1. Evaluation

- You can evaluate the performance of our models or models trained by yourself.

- Our models are at <https://1drv.ms/u/s!Aof3zM2LS8h6k1LNJ-1TKYSTIzH3?e=GCiZID>, please download the checkpoints to `checkpoints/`

  ```bash
  # evaluate teachers
  python eval_teacher.py --teacher resnet110 
  python eval_teacher.py --teacher resnet56 
  python eval_teacher.py --teacher wrn-40-2 
  python eval_teacher.py --teacher resnet32x4  
  # evaluate students
  python eval.py --model wrn-40-1 --teacher wrn-40-2 
  python eval.py --model wrn-16-2 --teacher wrn-40-2 
  python eval.py --model resnet20 --teacher resnet56 
  python eval.py --model resnet32 --teacher resnet110 
  python eval.py --model resnet8x4 --teacher resnet32x4 
  ```

2. Training on CIFAR-100

- Download the `teacher_ckpt.zip` at <https://1drv.ms/u/s!Aof3zM2LS8h6k1OWkXoUZANjbm3Y?e=TmWfE1> and unzip it to `teacher_ckpt/`.
- Download the CIAFR-100 dataset to `CIFAR-100/`
  ```bash
  # To train on TSKD.
  python train.py  --model wrn-16-2 --teacher wrn-40-2  --teacher-weight teacher_ckpt/cifar100_wrn-40-2__baseline1_best.pt --kd-loss-weight 5.0 --suffix lstm_review_kd
  python train.py  --model wrn-40-1 --teacher wrn-40-2  --teacher-weight teacher_ckpt/cifar100_wrn-40-2__baseline1_best.pt --kd-loss-weight 5.0 --suffix lstm_review_kd
  python train.py  --model resnet20 --teacher resnet56  --teacher-weight teacher_ckpt/cifar100_resnet56__baseline1_best.pt --kd-loss-weight 5.0 --suffix lstm_review_kd
  python train.py  --model resnet32 --teacher resnet110  --teacher-weight teacher_ckpt/cifar100_resnet110__baseline1_best.pt --kd-loss-weight 5.0 --suffix lstm_review_kd
  python train.py  --model resnet8x4 --teacher resnet32x4  --teacher-weight teacher_ckpt/cifar100_resnet32x4__baseline1_best.pt --kd-loss-weight 5.0 --suffix lstm_review_kd

  ```

# Acknowledgement

- Thanks for Conv-LSTM, DKD and ReviewKD. We build this library based on the [ConvLSTM](https://github.com/Hzzone/Precipitation-Nowcasting), 
[DKD's codebase](https://github.com/megvii-research/mdistiller) and the [ReviewKD's codebase](https://github.com/dvlab-research/ReviewKD).


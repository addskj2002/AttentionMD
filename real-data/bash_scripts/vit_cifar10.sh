#!/bin/bash

optim="Adam"
dataset="cifar10"
depth=6
heads=8
dim=512
filename="out_Adam_cosine_cifar10.pth"
lr=1e-4
epochs=2000
python train_vit.py --scheduler "cosine" --optim $optim --dataset $dataset --depth $depth --heads $heads --dim $dim --filename $filename --lr $lr --epochs $epochs

optim="1.1"
dataset="cifar10"
depth=6
heads=8
dim=512
filename="out_1.1_cosine_cifar10.pth"
lr=1e-1
epochs=2000
python train_vit.py --scheduler "cosine" --optim $optim --dataset $dataset --depth $depth --heads $heads --dim $dim --filename $filename --lr $lr --epochs $epochs

optim="1.75"
dataset="cifar10"
depth=6
heads=8
dim=512
filename="out_1.75_cosine_cifar10.pth"
lr=1e-1
epochs=2000
python train_vit.py --scheduler "cosine" --optim $optim --dataset $dataset --depth $depth --heads $heads --dim $dim --filename $filename --lr $lr --epochs $epochs

optim="2"
dataset="cifar10"
depth=6
heads=8
dim=512
filename="out_2_cosine_cifar10.pth"
lr=1e-1
epochs=2000
python train_vit.py --scheduler "cosine" --optim $optim --dataset $dataset --depth $depth --heads $heads --dim $dim --filename $filename --lr $lr --epochs $epochs

optim="3"
dataset="cifar10"
depth=6
heads=8
dim=512
filename="out_3_cosine_cifar10.pth"
lr=1e-2
epochs=2000
python train_vit.py --scheduler "cosine" --optim $optim --dataset $dataset --depth $depth --heads $heads --dim $dim --filename $filename --lr $lr --epochs $epochs

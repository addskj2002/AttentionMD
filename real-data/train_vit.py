import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from tools.vit_model import ViT
from tools.optimizer import SMD

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--optim', type=str, help='optimizer')
parser.add_argument('--scheduler', type=str, help='scheduler type')
parser.add_argument('--dataset', type=str, help='the dataset to use')
parser.add_argument('--depth', type=int, help='neural network depth')
parser.add_argument('--heads', type=int, help='number of heads in attention')
parser.add_argument('--dim', type=int, help='hidden layer dimension')
parser.add_argument('--original', default=None, type=str, help='default checkpoint')
parser.add_argument('--filename', type=str, help='outfile destination')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

# Dataset and dataloaders
if args.dataset == "cifar10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    num_classes = 10
elif args.dataset == "cifar100":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    num_classes = 100
print("Got dataloader")

# Model
if args.original is not None:
    checkpoint = torch.load(args.original, weights_only=False, map_location=device)
    model = checkpoint["model"].to(device)
else:
    model = ViT(
        image_size = 48,
        patch_size = 4,
        num_classes = num_classes,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.dim,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)
    checkpoint = {
        "train_losses": [],
        "train_accuracies": [],
        "test_losses": [],
        "test_accuracies": [],
        "time_taken": [],
    }
print("Got model")

# Optimizer
optimizer = SMD(
    [{"params": list(model.parameters()), "lr": args.lr}], p=float(args.optim)
) if args.optim != "Adam" and args.optim != "2" else (
    optim.Adam(model.parameters(), lr=args.lr)
    if args.optim == "Adam" else optim.SGD(model.parameters(), lr=args.lr)
)
loss_fn = nn.CrossEntropyLoss().to(device)
print("Got optimizer and loss fn")

# Scheduler
if args.scheduler == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


for idx in range(1, args.epochs + 1):
    print()
    print(f"Epoch {idx}")
    # Train
    model.train()
    if args.scheduler == "cosine":
        scheduler.step(idx-1)
        print(optimizer.param_groups[0]["lr"])
        # print(scheduler.get_lr())
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for inputs, targets in tqdm(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    avg_train_loss = train_loss / total
    avg_train_acc = correct / total
    checkpoint["train_losses"].append(avg_train_loss)
    checkpoint["train_accuracies"].append(avg_train_acc)
    checkpoint["time_taken"].append(end_time - start_time)
    print(f"Training Loss = {avg_train_loss}, Training Accuracy = {avg_train_acc}")

    # Test
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        avg_test_loss = test_loss / total
        avg_test_acc = correct / total
    checkpoint["test_losses"].append(avg_test_loss)
    checkpoint["test_accuracies"].append(avg_test_acc)
    print(f"Testing Loss = {avg_test_loss}, Testing Accuracy = {avg_test_acc}")

    # Save
    checkpoint["model"] = model
    torch.save(checkpoint, args.filename)

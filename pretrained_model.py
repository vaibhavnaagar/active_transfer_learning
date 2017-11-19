from __future__ import print_function
import os, sys
import argparse
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Pretrained Model fetaures')
parser.add_argument('--dataset', '-d', required=True, help='cifar10 | cifar100')
parser.add_argument('--model', '-m', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--batchsize', '-b', type=int, default=100, help='batchsize')
parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

args = parser.parse_args()

use_cuda = args.cuda and torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.__dict__[args.dataset.upper()](root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)

testset = torchvision.datasets.__dict__[args.dataset.upper()](root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

net = models.__dict__[args.model](pretrained=True)
# print(net)

class NetFeatures(nn.Module):
    def __init__(self, original_model):
        super(NetFeatures, self).__init__()
        # print(list(original_model.children())[:-1])
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        ## For PreActResnet only ##
        # x = F.avg_pool2d(x, 4)
        # x = x.view(x.size(0), -1)
        return x

# print(list(net.children())[:-1])
if args.model == "alexnet":
    new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
    net.classifier = new_classifier
else:
    net = NetFeatures(net)
print(net)

if use_cuda:
    net.cuda()
    # net_features.cuda()
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=range(ngpu))
        cudnn.benchmark = True

def run_model(net, loader):
    net.eval()
    embeddings = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        print("%d/%d" % (batch_idx, len(loader)))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        embeddings.append(outputs.cpu().data.numpy())
        labels.append(targets.cpu().data.numpy().reshape(-1,1))
    embeddings = np.vstack(tuple(embeddings))
    labels = np.vstack(tuple(labels))
    return embeddings, labels

train_embeddings, train_labels = run_model(net, trainloader)
test_embeddings, test_labels = run_model(net, testloader)
embeddings = dict(train_features=train_embeddings, train_labels=train_labels,
                  test_features=test_embeddings, test_labels=test_labels)
print(train_embeddings.shape, train_labels.shape)
print(test_embeddings.shape, test_labels.shape)
with open('%s_%s_features.pkl' % (args.dataset, args.model), 'wb') as f:
    pickle.dump(embeddings, f)

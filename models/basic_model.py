from __future__ import print_function, division
import torch
import torchvision
import numpy as np
import os

cifar_directory = './data/CIFAR/'

data_loader = torch.utils.data.DataLoader(os.path.join(cifar_directory, 'train'),
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
#         self.model = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75), # 0.75
            nn.Linear(in_features=25088, out_features=num_classes), # 25088
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
#         out = self.model(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class ResNet50(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(ResNet50, self).__init__()
        self.features = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=1000, out_features=num_classes),
            nn.Softmax()
            )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def single_emd_loss(p, q, num,r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
#     num = 1.
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += (sum(torch.abs(p[:i] - q[:i])) ** r) / num
#          emd_loss += (sum(torch.abs(p[:i] - q[:i])) ** r)
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, num, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], num=num[i].item(), r=r))
#          loss_vector.append(single_emd_loss(p[i], q[i], r=r))
#     return sum(loss_vector) / mini_batch_size

def cross_entropy_loss(p,q):
    """
    Cross Entropy of two arrays
    """
    assert p.shape == q.shape, 'Shape of the two distribution batches must be the same.'
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(-p[i]*np.log(q[i]))
    return sum(loss_vector) / mini_batch_size
    

def single_loss(p,q,num):
    """
    Naive mean square error
    """
    assert p.shape == q.shape, 'Shape of the two distribution batches must be the same.'
    length = p.shape[0]
    loss = 0.0
#     num = 1
    for i in range(0, length):
        loss += sum((p[i]-q[i])**2) / num
    return (loss / length) ** (1. / 2)

def naive_loss(p,q,num):
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_loss(p[i], q[i], num=num[i].item()))
    return sum(loss_vector) / mini_batch_size


def single_binary_loss(p,q,num):
    assert p.shape == q.shape, 'Shape of the two distribution batches must be the same'
    loss = 0.0
#     for i in range(0,length):
    loss += ((sum(torch.abs(p[:5]-q[:5])))**2 + (sum(torch.abs(p[5:10]-q[5:10])))**2) * num
    return (loss / 2) ** (1/2.)

def binary_loss(p,q,num):
    assert p.shape == q.shape, 'Shape of the two distribution batches must be the same'
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_binary_loss(p[i], q[i], num=num[i].item()))
    return sum(loss_vector) / mini_batch_size
    
# def single_emd_loss(p, q, r=2):
#     assert p.shape == q.shape, "Length of the two distribution must be the same"
#     length = p.shape[0]
#     emd_loss = 0.0
#     for i in range(1, length + 1):
#         emd_loss += sum(torch.abs(p[:i] - q[:i])) ** r
#     return (emd_loss / length) ** (1. / r)
#
# def emd_loss(p, q, r=2):
#     assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
#     mini_batch_size = p.shape[0]
#     loss_vector = []
#     for i in range(mini_batch_size):
#         loss_vector.append(single_emd_loss(p[i], q[i], r=r))
#     return sum(loss_vector) / mini_batch_size




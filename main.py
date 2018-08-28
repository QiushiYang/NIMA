# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import lrs

from data_loader import AVADataset
from tensorboardX import SummaryWriter
from model import *
from utils import *
from scipy.stats import pearsonr 
from scipy.stats import spearmanr 



def acc1(pred,label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist = torch.arange(10).float().to(device)
    p_mean = (pred.view(-1, 10) * dist).sum(dim=1)
    l_mean = (label.view(-1, 10) * dist).sum(dim=1)
    p_good = p_mean > 5
    l_good = l_mean > 5
    acc = (p_good == l_good).float().mean()
    return acc


def acc(pred,label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist = torch.arange(1,11).float().to(device)
    p_mean = (pred.view(-1, 10) * dist).sum(dim=1)
    l_mean = (label.view(-1, 10) * dist).sum(dim=1)
    p_good = p_mean > 5
    l_good = l_mean > 5
    acc = (p_good == l_good).float().mean()
    return acc

def main(config):

    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        
    ])

#     trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, style_file=config.style_ann_file, transform=train_transform)
#     valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, style_file=config.style_ann_file, transform=val_transform)
    
    trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, transform=train_transform)
    valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
        shuffle=True, num_workers=config.num_workers)

#     base_model = models.vgg16(pretrained=True)
    base_model = models.resnet50(pretrained=True)
    model = ResNet50(base_model)
    model = model.to(device)
    
#     base_model = models.inception_v3(pretrained=True)
#     base_model = models.resnet50(pretrained=True)
#     model = NIMA(base_model)

    if config.warm_start:
        model.load_state_dict(torch.load('./ckpts/epoch-%d.pkl' % config.warm_start_epoch))
        print('Successfully loaded model epoch-%d.pkl' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
                {'params': model.parameters(), 'lr': dense_lr}],
                momentum=0.9
                )
#     optimizer = optim.Adam([
#         {'params': model.features.parameters(), 'lr': conv_base_lr},
#         {'params': model.classifier.parameters(), 'lr': dense_lr}],
#         betas=(0.9, 0.99))

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc  =[]
        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            batch_acc = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                nums= data['number'].to(device).float()
                nums = nums ** (1/2)         
                
                outputs = model(images)                
                outputs = outputs.view(-1, 10, 1)                
                optimizer.zero_grad()   
        
                accuracy = acc(labels,outputs)
                accuracy1 = acc1(labels,outputs)
                
                batch_acc.append(accuracy.item())                
                loss = emd_loss(labels, outputs, nums)
#                 loss = emd_loss(labels, outputs)
#                 loss = naive_loss(labels, outputs,nums)
#                 loss = binary_loss(labels,outputs,nums)
                batch_losses.append(loss.item())                
                loss.backward()               
                optimizer.step()
                print('Epoch: %d/%d | Step: %d/%d | Training EMDLoss: %.4f | Accuracy : %.4f | %.4f ' %
(epoch + 1, config.epochs, i + 1, len(trainset)//config.train_batch_size + 1, loss.data[0],accuracy.data[0],accuracy1.data[0]))

                writer.add_scalar('datas/train_loss', loss.data[0], i+epoch*(len(trainset) // config.train_batch_size + 1))
                writer.add_scalar('datas/train_acc', accuracy.data[0], i+epoch*(len(trainset) // config.train_batch_size + 1))
                if (i % 500 == 0) and (i != 0):         
                    print('Saving model...')
                    torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'EMDLoss_noweight_resnet__epoch-%d-%d.pkl' % (epoch + 1,i)))
                    print('Done.\n')
                if (i % 99 == 0) and (i != 0):                    
                    batch_val_losses = [] 
                    batch_val_acc = []
                    batch_val_acc1 = []                   
#                      valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path,             style_file=config.style_ann_file, transform=val_transform) 
                    valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path,           transform=val_transform) 
                    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
                                                         shuffle=True, num_workers=config.num_workers)            
    #             for data in val_loader:
                    for j, data in enumerate(val_loader):
                        if j < 20:
                            images = data['image'].to(device)
                            labels = data['annotations'].to(device).float()
                            nums= data['number'].to(device).float()
                            nums = nums ** (1/2)
            #                print (labels)               
                            with torch.no_grad():
                                outputs = model(images)                                    
                            outputs = outputs.view(-1, 10, 1) 

                            val_accuracy = acc(labels,outputs)
                            val_accuracy1 = acc1(labels,outputs)                           
                            
                            batch_val_acc.append(val_accuracy.item())
                            batch_val_acc1.append(val_accuracy1.item())
                            val_loss = emd_loss(labels, outputs, nums)
#                             val_loss = naive_loss(labels, outputs,nums)
#                             val_loss = binary_loss(labels,outputs,nums)
#                             batch_val_losses.append(val_loss.item()) 

#                     avg_val_acc = sum(batch_val_acc) / (len(valset) // config.val_batch_size + 1)
                    avg_val_acc = sum(batch_val_acc) / 20
                    avg_val_acc1 = sum(batch_val_acc1) / 20                                 
    
                    val_acc.append(avg_val_acc)
#                     avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
                    avg_val_loss = sum(batch_val_losses) / 20
#                     val_losses.append(avg_val_loss)
                    print('Epoch %d / Iters %d completed.| Accuracy on val set: %.4f | %.4f ' % (epoch + 1,i,avg_val_acc,avg_val_acc1))
                    writer.add_scalar('datas/val_loss', avg_val_loss, i*(epoch+1))
                    writer.add_scalar('datas/val_acc', avg_val_acc, i*(epoch+1))

#             avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
#             train_losses.append(avg_loss)
#             print('Epoch %d averaged training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9
                )

#             # Use early stopping to monitor training
#             if avg_val_loss < init_val_loss:
#                 init_val_loss = avg_val_loss
#                 # save model weights if val loss decreases
#                 print('Saving model...')
#                 torch.save(model.state_dict(), os.path.join(config.ckpt_path, '/epoch-%d.pkl' % (epoch + 1)))
#                 print('Done.\n')
#                 # reset count
#                 count = 0
#             elif avg_val_loss >= init_val_loss:
#                 count += 1
#                 if count == config.early_stopping_patience:
#                     print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
#                     break

        print('Training completed.')
        writer.close()

        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')
    
    if config.test:
        model.load_state_dict(torch.load('./ckpts/CorLoss_noweight_resnet__epoch-2-3100.pkl'))
#         torch.load('./epoch-10.pkl')
        model.eval()
        # compute mean score
        test_transform = val_transform
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.test_img_path, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=True, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        batch_test_acc = []
        
        out_mean = []
        out_std = []
        label_mean = []
        label_std = []
#         print('Successfully loaded model epoch-%d.pkl' % config.warm_start_epoch)
#         for data in test_loader:
        for j, data in enumerate(test_loader):
#             if j < 500:
                image = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                labels = labels.view(-1,10)
                nums= data['number'].to(device)
                nums = nums ** (1/2)

                labels = F.softmax(labels, dim=1)
                output = model(image)
                output = output.view(-1, 10, 1)
                output = output.view(-1, 10)

                out_mean.append(np.array((output*torch.range(1,10).to(device)).mean(1).item()))
                out_std.append(np.array((output*torch.range(1,10).to(device)).std(1).item()))
                label_mean.append(np.array((labels*torch.range(1,10).to(device)).mean(1).item()))
                label_std.append(np.array((labels*torch.range(1,10).to(device)).std(1).item()))

                test_accuracy = acc(labels,output)
                batch_test_acc.append(test_accuracy)

    #             loss = emd_loss(labels, output,nums)
    #             print ('loss: ' + str(loss.item()))
#                 predicted_mean, predicted_std = 0.0, 0.0
#                 for i, elem in enumerate(output, 1):
#                     predicted_mean += i * elem
#                 for j, elem in enumerate(output, 1):
#                     predicted_std += elem * (i - predicted_mean) ** 2
#                 mean_preds.append(predicted_mean)
#                 std_preds.append(predicted_std)
        
#         avg_test_loss = sum(batch_test_losses) / (len(testset) // config.test_batch_size + 1)
#         test_losses.append(avg_test_loss)
        avg_test_acc = sum(batch_test_acc) / (len(testset) // config.test_batch_size + 1)
#         avg_test_acc = sum(batch_test_acc) / 500

        lcc_mean = LCC(out_mean,label_mean)
        lcc_std  =LCC(out_std,label_std)
        srcc_mean = SRCC(out_mean,label_mean)
        srcc_std = SRCC(out_std,label_std)
        print('Completed. Averaged Accuracy on test set: %.4f | LCC: mean: %.4f, std:  %.4f | SRCC:mean: %.4f, std:%.4f ' % (avg_test_acc, lcc_mean,lcc_std,srcc_mean,srcc_std))
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/data/full_ava/train/images')
    parser.add_argument('--val_img_path', type=str, default='/data/full_ava/train/images')
    parser.add_argument('--test_img_path', type=str, default='/data/full_ava/train/images')
#     parser.add_argument('--style_ann_file', type=str, default='./style_file.txt')
    
    parser.add_argument('--train_csv_file', type=str, default='./train.txt')
#     parser.add_argument('--train_csv_file', type=str, default='./all_ann.txt')
    parser.add_argument('--val_csv_file', type=str, default='./val.txt')
    parser.add_argument('--test_csv_file', type=str, default='./small_test.txt')

    # training parameters
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=3e-3)  # 3e-7
    parser.add_argument('--dense_lr', type=float, default=3e-2)  # 3e-6
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=64)  # 128
    parser.add_argument('--val_batch_size', type=int, default=64)    # 128
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)  # 2

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./ckpts')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)  # False
    parser.add_argument('--warm_start_epoch', type=int, default=0) # 0
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', type=bool, default=True)

    config = parser.parse_args()

    main(config)


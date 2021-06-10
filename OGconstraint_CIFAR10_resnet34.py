import torch
import torch.nn as nn
import numpy as np
from models import *
from Optimizers import OGconstraint_ud
from Optimizers import initOGconstraint
from datasets import CIFAR10data
from train import train
from test import test

torch.cuda.set_device(2)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}.")
torch.manual_seed(5) #optional

#Hyperparameters
h = 0.1
T = 0
dt1 = h/3 #for warm-up
cgamma = 0.9 
WD = 0 
dgamma = 0
num_runs = 3
num_epochs = 150
batchsize = 128

loader_train,loader_test = CIFAR10data.generatedata(batchsize=batchsize)

RES_train_loss_allruns = []
RES_test_loss_allruns = []
RES_test_acc_allruns = []
RES_train_acc_allruns = []

for run in range(num_runs):
    print("run = ", run)
    net = ResNet34()
    Constrainedlist, net = initOGconstraint.initOG(net)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = OGconstraint_ud.oCoLAud(net.parameters(),device,Constrainedlist=Constrainedlist,lr=dt1,cgamma=cgamma,dgamma=dgamma,weight_decay=WD) 

    RES_train_loss = []
    RES_train_acc = []
    RES_test_loss = []
    RES_test_acc = []

    for epoch in range(num_epochs):

        net, optimizer, loss_train,acc_train = train(epoch,loader_train,net,optimizer,criterion,device)
        loss_test,acc_test = test(loader_test,net,criterion,device)
            
        RES_train_loss.append(loss_train)
        RES_train_acc.append(acc_train)
        RES_test_loss.append(loss_test)
        RES_test_acc.append(acc_test)
        
        #warmup
        if epoch < 2: 
            dt1 += (h/3) 
            optimizer.param_groups[0]['lr'] = dt1
        #learning rate decay
        elif epoch == 50:
            optimizer.param_groups[0]['lr'] = 0.01
        elif epoch == 100:
            optimizer.param_groups[0]['lr'] = 0.001 


    RES_train_loss_allruns.append(RES_train_loss)
    RES_train_acc_allruns.append(RES_train_acc)
    RES_test_loss_allruns.append(RES_test_loss)
    RES_test_acc_allruns.append(RES_test_acc)


with open(f'OGconstraint_Resnet34_CIFAR10_batchsize_{batchsize}_WD_{WD}_cgam_{cgamma}_h_{h}_T_{T}_{num_runs}runs_{num_epochs}epochs.txt', 'w+') as f:
    f.write(f'Training loss min: {np.min(RES_train_loss_allruns,0)}\n') 
    f.write(f'Test loss min: {np.min(RES_test_loss_allruns,0)}\n') 
    f.write(f'Training accuracy min: {np.min(RES_train_acc_allruns,0)}\n') 
    f.write(f'Test accuracy min: {np.min(RES_test_acc_allruns,0)}\n') 
    f.write(f'Training loss max: {np.max(RES_train_loss_allruns,0)}\n') 
    f.write(f'Test loss max: {np.max(RES_test_loss_allruns,0)}\n') 
    f.write(f'Training accuracy max: {np.max(RES_train_acc_allruns,0)}\n') 
    f.write(f'Test accuracy max: {np.max(RES_test_acc_allruns,0)}\n') 
    f.write(f'Training loss std: {np.std(RES_train_loss_allruns,0)}\n') 
    f.write(f'Test loss std: {np.std(RES_test_loss_allruns,0)}\n') 
    f.write(f'Training accuracy std: {np.std(RES_train_acc_allruns,0)}\n') 
    f.write(f'Test accuracy std: {np.std(RES_test_acc_allruns,0)}\n') 
    f.write(f'Training loss mean: {np.mean(RES_train_loss_allruns,0)}\n') 
    f.write(f'Test loss mean: {np.mean(RES_test_loss_allruns,0)}\n') 
    f.write(f'Training accuracy mean: {np.mean(RES_train_acc_allruns,0)}\n') 
    f.write(f'Test accuracy mean: {np.mean(RES_test_acc_allruns,0)}\n') 



import numpy as np
import torch
from torch import nn
from datasets import FashionMNISTdata
from Optimizers import circleconstraint_ud
from train import train
from test import test

# Run on cuda
torch.cuda.set_device(3)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}.")

# Hyperparameter settings
h    = 0.3
gam  = 1
rad0 = 0.05
rad1 = 0.1
T    = 0
print("h = ", h, ", rad0 =", rad0, ", rad1 =", rad1, ",gam = ", gam)
num_epochs = 400
num_runs   = 20 
batchsize  = 128

cgamma = np.exp(-h*gam)
if T == 0:
    dgamma = 0
else:
    dgamma = np.sqrt(1-np.exp(-2*h*gam))*np.sqrt(T)
torch.manual_seed(2) #optional

loader_train,loader_test = FashionMNISTdata.generatedata(batchsize=batchsize)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
                                            
        self.passthrough = nn.Sequential(nn.Linear(784,1000), 
                                      nn.ReLU(), 
                                      nn.Linear(1000,10)             
                                      )
    
    def forward(self, x):
        out = self.passthrough(x)
        return out 

NN = Net()
biaslist = []
for name, param in NN.named_parameters(): 
    if "bias" not in name:
        biaslist.append(0)
    else:
        biaslist.append(1)
biaslist = np.hstack(biaslist)

RES_train_loss_allruns = []
RES_test_loss_allruns = []
RES_test_acc_allruns = []
RES_train_acc_allruns = []

for run in range(num_runs):
    print("run", run)
    NN = Net()
    NN = NN.to(device)

    optimizer = circleconstraint_ud.cCoLAud(NN.parameters(),biaslist=biaslist,lr=h,rad0=rad0,rad1=rad1,cgam=cgamma,dgam=dgamma)
    criterion = nn.CrossEntropyLoss()

    RES_train_loss = []
    RES_test_loss = []
    RES_test_acc = []
    RES_train_acc = []

    for epoch in range(num_epochs): 

        NN, optimizer, loss_train,acc_train = train(epoch,loader_train,NN,optimizer,criterion,device,784)
        loss_test,acc_test = test(loader_test,NN,criterion,device,784)

        if epoch % 5 == 0:
            RES_train_loss.append(loss_train)
            RES_train_acc.append(acc_train)
            RES_test_loss.append(loss_test)
            RES_test_acc.append(acc_test)
             
    RES_train_loss_allruns.append(RES_train_loss)
    RES_train_acc_allruns.append(RES_train_acc)
    RES_test_loss_allruns.append(RES_test_loss)
    RES_test_acc_allruns.append(RES_test_acc)


with open(f'Circleconstraint_FashionMNIST_{num_epochs}epochs_hyperparameters_batchsize_{batchsize}_h_{h}_T_{T}_rad0_{rad0}_rad1_{rad1}_gamma_{gam}_{num_runs}runs.txt', 'w+') as f:
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


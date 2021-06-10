import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np

class oCoLAud(Optimizer):
    def __init__(self,params,device,Constrainedlist,lr=0.1,cgamma=0,dgamma=0,weight_decay=0): 
        self.device = device
        self.Constrainedlist = Constrainedlist
        defaults = dict(lr=lr,cgamma=cgamma,dgamma=dgamma,weight_decay=weight_decay)
        super(oCoLAud,self).__init__(params,defaults)
           
    def __setstate__(self,state):
        super(oCoLAud,self).__setstate__(state)

    @torch.no_grad()
    def stepMom(self):
        for group in self.param_groups:
            
            for i,p in enumerate(group['params']):
            
                if p.grad is None:
                    continue
            
                param_state =self.state[p]
                shapep = p.shape 
                if self.Constrainedlist[i] == 1:
                    shapep0 = shapep[0]
                    if len(shapep) > 2:
                        shapep1 = shapep[1]*shapep[2]*shapep[3]
                    else:
                        shapep1 = shapep[1]

                    d_p = p.grad
                    buf = param_state['momentum_buffer'] = -0.01*torch.clone(d_p).detach() 
                    buffy = torch.clone(buf).detach().reshape((shapep0,shapep1))
                    Weighty = torch.clone(p).detach().reshape((shapep0,shapep1))

                    if shapep0 >= shapep1:
                        bufproj = -0.5*torch.matmul(Weighty,(torch.matmul(torch.transpose(buffy,0,1),Weighty)+torch.matmul(torch.transpose(Weighty,0,1),buffy))).reshape(*shapep)
                    else:
                        bufproj = -0.5*torch.transpose(torch.matmul(torch.transpose(Weighty,0,1),(torch.matmul(Weighty,torch.transpose(buffy,0,1))+torch.matmul(buffy,torch.transpose(Weighty,0,1)))),0,1).reshape(*shapep)

                    buf.add_(bufproj)

                else:
                    d_p = p.grad
                    buf = param_state['momentum_buffer'] = -0.01*torch.clone(d_p).detach()


    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            cgamma = group['cgamma']
            dgamma = group['dgamma']
            weight_decay = group['weight_decay']

            for i,p in enumerate(group['params']):
                
                if p.grad is None:
                    continue

                param_state = self.state[p]
                shapep = p.shape

                if self.Constrainedlist[i] == 1:

                    shapep0 = shapep[0]
                    if len(shapep) > 2:
                        shapep1 = shapep[1]*shapep[2]*shapep[3]
                    else:
                        shapep1 = shapep[1] 

                    if 'OldWeight' not in param_state:
                        OldWeight = param_state['OldWeight'] = torch.clone(p).detach()
                        OldWeight = OldWeight.reshape((shapep0,shapep1))
                        if shapep0 >= shapep1:
                            prodis = torch.matmul(torch.transpose(OldWeight,0,1),OldWeight)
                        else:
                            prodis = torch.matmul(OldWeight,torch.transpose(OldWeight,0,1))
                            OldWeightT = torch.transpose(OldWeight,0,1)
                        Id = param_state['Id'] = torch.eye(*prodis.shape).to(self.device)
                    else:
                        OldWeight = param_state['OldWeight']
                        OldWeight = torch.clone(p).detach()
                        OldWeight = OldWeight.reshape((shapep0,shapep1))
                        if shapep0 < shapep1:
                            OldWeightT = torch.transpose(OldWeight,0,1)
                        Id = param_state['Id']

                    buf = param_state['momentum_buffer'] 

                    # O -step
                    if dgamma == 0:
                        buf.mul_(cgamma)
                    else:
                        buf.mul_(cgamma).add_(dgamma,torch.cuda.FloatTensor(*shapep).normal_())
                    buffy = torch.clone(buf).detach().reshape((shapep0,shapep1))
                    if shapep0 >= shapep1:
                        bufproj = -0.5*torch.matmul(OldWeight,(torch.matmul(torch.transpose(buffy,0,1),OldWeight)+torch.matmul(torch.transpose(OldWeight,0,1),buffy))).reshape(*shapep)
                    else:
                        bufproj = -0.5*torch.transpose(torch.matmul(OldWeightT,(torch.matmul(OldWeight,torch.transpose(buffy,0,1))+torch.matmul(buffy,torch.transpose(OldWeight,0,1)))),0,1).reshape(*shapep)
                    
                    buf.add_(bufproj)

                    # B-step
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)

                    buf.add_(-d_p) 
                    buffy = torch.clone(buf).detach().reshape((shapep0,shapep1))
                    if shapep0 >= shapep1:
                        bufproj = -0.5*torch.matmul(OldWeight,(torch.matmul(torch.transpose(buffy,0,1),OldWeight)+torch.matmul(torch.transpose(OldWeight,0,1),buffy))).reshape(*shapep)
                    else:
                        bufproj = -0.5*torch.transpose(torch.matmul(OldWeightT,(torch.matmul(OldWeight,torch.transpose(buffy,0,1))+torch.matmul(buffy,torch.transpose(OldWeight,0,1)))),0,1).reshape(*shapep)
                    
                    
                    buf.add_(bufproj)
                    d_p = buf

                    # A-step
                    p.data.add_(d_p,alpha=group['lr'])
                    p.data = p.reshape((shapep0,shapep1))
                    FirstStep = torch.clone(p).detach()

                    if shapep0 >= shapep1:
                        for ks in range(10):
                            Lambda = torch.matmul(torch.transpose(p,0,1),p)-Id
                            products = -0.5*torch.matmul(OldWeight,Lambda)
                            p.add_(products)

                        bufproj1 = ((p.data-FirstStep)/group['lr']).reshape(*shapep)
                        buf.add_(bufproj1)
                    else:
                        for ks in range(10):
                            Lambda = torch.matmul(p,torch.transpose(p,0,1))-Id
                            products = -0.5*torch.transpose(torch.matmul(OldWeightT,Lambda),0,1)
                            p.add_(products)

                        bufproj1 = ((p.data-FirstStep)/group['lr']).reshape(*shapep)
                        buf.add_(bufproj1)

                    p.data = p.reshape(*shapep)
                     
                    OldWeight = torch.clone(p).detach()
                    OldWeight = OldWeight.reshape((shapep0,shapep1))

                    buffy = torch.clone(buf).detach().reshape((shapep0,shapep1))
                    if shapep0 >= shapep1:
                        bufproj = -0.5*torch.matmul(OldWeight,(torch.matmul(torch.transpose(buffy,0,1),OldWeight)+torch.matmul(torch.transpose(OldWeight,0,1),buffy))).reshape(*shapep)
                    else:
                        bufproj = -0.5*torch.transpose(torch.matmul(OldWeightT,(torch.matmul(OldWeight,torch.transpose(buffy,0,1))+torch.matmul(buffy,torch.transpose(OldWeight,0,1)))),0,1).reshape(*shapep)
                    
                    buf.add_(bufproj)
                else:
                    buf = param_state['momentum_buffer'] 

                    if dgamma == 0:
                        buf.mul_(cgamma)
                    else:
                        buf.mul_(cgamma).add_(dgamma,torch.cuda.FloatTensor(*shapep).normal_())
                    
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)

                    buf.add_(-d_p,alpha=1) 
                    d_p = buf
                    p.data.add_(d_p,alpha=group['lr'])







        

               

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

class cCoLAud(Optimizer):
    def __init__(self,params,biaslist,lr=0.1,rad0=0.05,rad1=0.05,cgam=0,dgam=0):
        self.biaslist = biaslist
        defaults = dict(lr=lr,rad0=rad0,rad1=rad1,cgam=cgam,dgam=dgam)
        super(cCoLAud,self).__init__(params,defaults)
           
    def __setstate__(self,state):
        super(cCoLAud,self).__setstate__(state)

    @torch.no_grad()
    def stepMom(self):
        for group in self.param_groups:
            rad0 = group['rad0']
            rad1 = group['rad1']

            for i,p in enumerate(group['params']):
            
                if p.grad is None:
                    continue
            
                param_state =self.state[p]

                if self.biaslist[i] == 0:
                    d_p = p.grad
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    xi_buf = param_state['xi_buffer'] = torch.clone(d_p).detach()
                    if i == 0:
                        rad = rad0
                    else:
                        rad = rad1
                    xi = param_state['xi'] = torch.sqrt(torch.abs(rad**2-p.data**2))
                    proj = p.data.mul(p.data.mul(buf)+xi.data.mul(xi_buf))
                    buf.add_(-(1/(rad**2)),proj)

                    projxi = xi.data.mul(p.data.mul(buf)+xi.data.mul(xi_buf))
                    xi_buf.add_(-(1/(rad**2)),projxi)

                    buf.add_(-group['lr']/2,d_p.mul(1-(1/(rad**2))*p.data**2))
                    xi_buf.add_(group['lr']/2,d_p.mul(xi).mul((1/(rad**2))*p.data))
                else:
                    d_p = p.grad
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    buf.add_(-group['lr']/2,d_p)


    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            cgam = group['cgam']
            dgam = group['dgam']
            rad0 = group['rad0']
            rad1 = group['rad1']

            for i,p in enumerate(group['params']):
                param_state = self.state[p]
                
                if self.biaslist[i] == 0:
                    buf = param_state['momentum_buffer'] 
                    xi = param_state['xi']  
                    xi_buf = param_state['xi_buffer']

                    if i == 0:
                        rad = rad0
                    else:
                        rad = rad1
                    
                    omega = (1/(rad**2))*torch.nn.Parameter(xi.mul(buf)-p.data.mul(xi_buf))
                    buf.data = omega.mul(xi.mul(torch.cos(0.5*omega*group['lr']))-p.data.mul(torch.sin(omega*0.5*group['lr'])))
                    xi_buf.data = -omega.mul(p.data.mul(torch.cos(0.5*omega*group['lr']))+xi.mul(torch.sin(0.5*omega*group['lr'])))

                    p.data.mul_(torch.cos(0.5*omega*group['lr'])).add_(xi.mul(torch.sin(0.5*omega*group['lr'])))
                    xi.mul_(torch.cos(0.5*omega*group['lr'])).add_(-p.data.mul(torch.sin(0.5*omega*group['lr'])))

                    shapep = p.size()

                    if dgam == 0:
                        buf.mul_(cgam)
                        xi_buf.mul_(cgam)
                    else:
                        buf.mul_(cgam).add_(dgam,torch.cuda.FloatTensor(*shapep).normal_())
                        xi_buf.mul_(cgam).add_(dgam,torch.cuda.FloatTensor(*shapep).normal_())

                    c = (1/(rad**2))*torch.nn.Parameter(p.data.mul(buf)+xi.mul(xi_buf))
                    buf.add_(-c.mul(p.data))
                    xi_buf.add_(-c.mul(xi))

                    omega = (1/(rad**2))*torch.nn.Parameter(xi.mul(buf)-p.data.mul(xi_buf))
                    buf.data = omega.mul(xi.mul(torch.cos(0.5*omega*group['lr']))-p.data.mul(torch.sin(omega*0.5*group['lr'])))
                    xi_buf.data = -omega.mul(p.data.mul(torch.cos(0.5*omega*group['lr']))+xi.mul(torch.sin(0.5*omega*group['lr'])))

                    p.data.mul_(torch.cos(0.5*omega*group['lr'])).add_(xi.mul(torch.sin(0.5*omega*group['lr'])))
                    xi.mul_(torch.cos(0.5*omega*group['lr'])).add_(-p.data.mul(torch.sin(0.5*omega*group['lr'])))

                    if p.grad is None:
                        continue
                
                    d_p = p.grad
                    
                    buf.add_(-group['lr'],d_p.mul(1-(1/(rad**2))*p.data**2))
                    xi_buf.add_(group['lr'],d_p.mul(xi).mul((1/(rad**2))*p.data))
                else:
                    buf = param_state['momentum_buffer'] 

                    p.data.add_(buf,alpha=group['lr']/2)

                    shapep = p.size()
                    if dgam == 0:
                        buf.mul_(cgam)
                    else:
                        buf.mul_(cgam).add_(dgam,torch.cuda.FloatTensor(*shapep).normal_())

                    p.data.add_(buf,alpha=group['lr']/2)

                    if p.grad is None:
                        continue
                
                    d_p = p.grad

                    buf.add_(-group['lr'],d_p)
                

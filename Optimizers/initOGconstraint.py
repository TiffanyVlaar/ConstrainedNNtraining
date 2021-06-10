import torch
import torch.nn as nn
import numpy as np

def initOG(net):
    Constrainedlist = []
    for i,named in enumerate(net.named_parameters()):
        name,pp = named
        shapep = pp.shape
       
        if i > 0 and len(shapep) >= 2: 
            shapep0 = shapep[0]
            if len(shapep)>2:
                shapep1 = shapep[1]*shapep[2]*shapep[3]
                w = torch.clone(pp.data).reshape(shapep0,shapep1)
            else:
                shapep1 = shapep[1]
                w = torch.clone(pp.data) 

            if shapep0 > shapep1:
                q, _ = np.linalg.qr(w)
            else:
                q, _ = np.linalg.qr(w.T)
            pp.data = torch.Tensor(q).reshape(*shapep)
            Constrainedlist.append(1)
        else:
            Constrainedlist.append(0)

    return Constrainedlist, net

        

               

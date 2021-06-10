def generatedata(batchsize):    
    import torch
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader, Dataset

    #Optional: 
    torch.manual_seed(2)

    mnist_trainold = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    loader_train_old = DataLoader(mnist_trainold, batch_size = int(len(mnist_trainold)/6), shuffle=True)
    mnist_testold = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader_test_old = DataLoader(mnist_testold, batch_size=len(mnist_testold), shuffle=True)


    class datal(Dataset):   
        def __init__(self,x,y):
            super().__init__()
            self.x = x
            self.y = y
            
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, i):
            return self.x[i], self.y[i]

    for i,data in enumerate(loader_test_old):
        xtest=data[0].view(-1,784)
        ytest=data[1] 

    for i,data in enumerate(loader_train_old):
        if i == 0:
            xtrain = data[0].view(-1,784)
            ytrain = data[1]
        else:
            xtest1 = data[0].view(-1,784)
            ytest1 = data[1] 

            xtest = torch.cat((xtest,xtest1))
            ytest = torch.cat((ytest,ytest1))

    train_new = datal(xtrain, ytrain)
    test_new = datal(xtest,ytest)  
    
    loader_train = DataLoader(train_new, batch_size=batchsize, shuffle=True)
    loader_test = DataLoader(test_new, batch_size=100, shuffle=False)

    return loader_train,loader_test


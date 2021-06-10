def generatedata(batchsize):    
    import torch
    import torchvision
    from torchvision import datasets,transforms

    #Optional: 
    torch.manual_seed(5)
        
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    loader_train = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    loader_test = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    return loader_train,loader_test


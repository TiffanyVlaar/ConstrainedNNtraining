def train(epoch,loader_train,NN,optimizer,criterion,device,inputshape=0): 
    NN.train()
    loss_train = 0
    correct = 0
    total_target = 0
    
    for i,data in enumerate(loader_train):

        if inputshape == 0:
            x = data[0].to(device) 
        else:
            x = data[0].view(-1,inputshape).to(device) 
            
        y = data[1].long().to(device) 

        if i == 0 and epoch == 0:
            output = NN(x)
            loss = criterion(output,y) 
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.stepMom()

        output = NN(x)
        loss = criterion(output,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        _, predict = output.max(1)
        total_target += y.shape[0]
        correct += predict.eq(y).sum().item()

    loss_train /= (i+1)
    acc_train = 100*(correct/total_target)
    
    print('\nEpoch: %d' % epoch)
    print('%d / %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'% (i+1, len(loader_train),loss_train, acc_train, correct, total_target))

    return NN, optimizer, loss_train,acc_train


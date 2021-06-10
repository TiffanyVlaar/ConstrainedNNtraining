import torch
def test(loader_test,NN,criterion,device,inputshape=0): 
    NN.eval()
    loss_test = 0
    correct = 0
    total_target = 0
    with torch.no_grad():
        for i,data in enumerate(loader_test):

            if inputshape == 0:
                x = data[0].to(device) 
            else:
                x = data[0].view(-1,inputshape).to(device) 
            y = data[1].long().to(device) 

            output = NN(x)
            loss = criterion(output,y)

            loss_test += loss.item()
            _, predict = output.max(1)
            total_target += y.shape[0]
            correct += predict.eq(y).sum().item()

        loss_test /= (i+1)
        acc_test = 100*(correct/total_target)
        print('Test loss: %.3f | Test acc: %.3f%% (%d/%d)'% (loss_test, acc_test, correct,total_target))

    return loss_test,acc_test


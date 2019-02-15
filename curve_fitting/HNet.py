import torch 
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):   
    def __init__(self,input_c,output_c,size,stride=3,padding=0):
        super(ConvBlock,self).__init__()
        self.net=nn.Sequential(nn.Conv2d(input_c,output_c,size,stride=stride,padding=padding),
                               nn.BatchNorm2d(output_c),
                               nn.ReLU())
    def forward(self,x):
        return self.net(x)

class LinearBlock(nn.Module):
    def __init__(self,input_c,output_c):
        super(LinearBlock,self).__init__()
        self.net=nn.Sequential(nn.Linear(input_c,output_c),
                               nn.BatchNorm2d(output_c),
                               nn.ReLU()) 
    def forward(self,x):
        return self.net(x)


class HNet(nn.Module):
    def __init__(self):
        super(HNet,self).__init__()
        self.conv=nn.Sequential(ConvBlock(3,16,3),
                                ConvBlock(16,16,3),
                                nn.AdaptiveMaxPool2d([64,32]),
                                ConvBlock(16,32,3),
                                ConvBlock(32,32,3),
                                nn.AdaptiveMaxPool2d([32,16]),
                                ConvBlock(32,64,3),
                                ConvBlock(64,64,3),
                                nn.AdaptiveMaxPool2d([16,8]))

        self.linear=nn.Sequential(LinearBlock(16*8*64,1024),
                                  nn.Linear(1024,6))

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size[0],-1)
        x=self.linear(x)
        return x

def hnet_loss(coefficient,data):
    transformation_matrix=torch.tensor([[coefficient[0,0],coefficient[0,1],coefficient[0,2]],
                                        [.0,coefficient[0,3],coefficient[0,4]],
                                        [.0,coefficient[0,5],.1]],dtype=torch.float)
    
    Loss=torch.tensor(0,dtype=torch.float).cuda()
    for k in data['lane'].keys():
        coordinates=torch.tensor(data['lane'][k],dtype=torch.float).cuda()
        projection=torch.matmul(transformation_matrix,coordinates)
        N=coordinates.size(0)

        Y_base=projection[:,0]
        Y_square=Y_base.pow(2)
        Y_const=torch.ones_like(Y_base)

        Y=torch.stack([Y_square,Y_base,Y_const],dim=-1)

        X=projection[1,:]
        X=X.unsqueeze(dim=-1)

        w=torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y.transpose(0,1),Y)),Y.transpose(0,1)),X)

        X_infered=torch_matmul(Y,w)

        loss=torch.sum((X_infered-X).pow(2))/N
        Loss+=loss
    return Loss

        
    


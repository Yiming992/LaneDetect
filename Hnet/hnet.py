import torch 
import torch.nn as nn
import numpy as np 

'''
-----------------------------
Part II: 深度学习车道线拟合网络

参考文献：
 "Towards End-to-End Lane Detection: an Instance Segmentation Approach",ArXiv2018,
 Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, Luc Van Gool
-----------------------------
'''
class ConvBlock(nn.Module):   
    def __init__(self,input_c,output_c,size,stride=3,padding=0):
        '''
        基础卷积模块
        变量:
            input_c:输入张量的通道数
            output_c:输出张量的通道数
            size:卷积核的尺寸
            stride:卷积核滑动的步长
            padding:zero-padding的数量
        '''
        super(ConvBlock,self).__init__()
        self.net=nn.Sequential(nn.Conv2d(input_c,output_c,size,stride=stride,padding=padding),
                               nn.BatchNorm2d(output_c),
                               nn.ReLU())
    def forward(self,x):
        '''
        定义基础卷积模块的前向传播方式
        '''
        return self.net(x)

class LinearBlock(nn.Module):
    def __init__(self,input_c,output_c):
        '''
        定义全连接线性模块
        变量：
           input_c:输入张量的通道数
           output_c:输出张量的通道数
        '''
        super(LinearBlock,self).__init__()
        self.net=nn.Sequential(nn.Linear(input_c,output_c),
                               nn.BatchNorm2d(output_c),
                               nn.ReLU()) 
    def forward(self,x):
        '''
        定义全连接模块的前向传播方式
        '''
        return self.net(x)


class HNet(nn.Module):
    def __init__(self):
        '''
        组合以上模块定义最终的HNet
        '''
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
        '''
        定义HNet前向传播方式
        '''
        x=self.conv(x)
        x=x.view(x.size[0],-1)
        x=self.linear(x)
        return x

'''
outputs:[batch,6]
labels:[batch,64,128]
'''
def hnet_loss(batch,outputs,labels,N):
    loss=torch.tensor(0.).cuda()
    for b in range(batch_size):
        label=labels[b,:,:]
        num_lane=len(label.unique())-1
        lane_values=label.unique()[1:]
        T_matrix=torch.tensor([[outputs[b,0],outputs[b,1],outputs[b,2]],[0.,outputs[b,3],outputs[b,4]],[0,outputs[b,5],1]])
        for l in range(num_lane):
            indices=(label==lane_values[l]).nonzero()
            sample_indices=indices[np.arange(0,indices.size()[0],5),:]
            sample_indices=torch.stack([[:,1],sample_indices[:,0]],dim=1)
            coordinates=torch.cat([sample_indices,torch.ones_like(sample_indices[:,0])],dim=1)
            
            x=coordinates[:,0]
            projection=torch.matmul(T_matrix,coordinates)

            x_prime=projection[:,0]
            y=projection[:,1]

            y_square=torch.pow(y,2)
            y_const=torch.ones_like(y)
            y=torch.stack([y_square,y,y_const],dim=1)
            
            coefficients=torch.matmul(torch.inverse(torch.matmul(torch.transpose(y),y)),torch.matmul(torch.transpose(y),x_prime))
            
            x_predicted=torch.matmul(y,coefficients)
                        
            loss+=torch.mean(torch.pow(x-x_predicted))
    return loss

    
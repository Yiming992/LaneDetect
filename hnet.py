import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,RandomSampler
from hnet_data import HnetData
import time
import os
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
                               nn.BatchNorm1d(output_c),
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
        x=x.view(x.size()[0],-1)
        x=self.linear(x)
        return x

'''
outputs:[batch,6]
labels:[batch,64,128]
'''
def hnet_loss(batch,outputs,labels):
    loss=torch.tensor(0.).cuda()
    for b in range(batch):
        label=labels[b,:,:]
        num_lane=len(label.unique())-1
        lane_values=label.unique()[1:]
        T_matrix=torch.stack([outputs[b,0],outputs[b,1],outputs[b,2],torch.tensor(0.).cuda(),outputs[b,3],outputs[b,4],torch.tensor(0.).cuda(),outputs[b,5],torch.tensor(1.).cuda()],dim=-1)
        T_matrix=T_matrix.view(3,3)
        #T_matrix=torch.tensor([[outputs[b,0],outputs[b,1],outputs[b,2]],[0.,outputs[b,3],outputs[b,4]],[0,outputs[b,5],1]]).cuda()
        for l in range(num_lane):
            indices=(label==lane_values[l]).nonzero()
            sample_indices=indices[np.arange(0,indices.size()[0],5),:]
            print(sample_indices)
            coordinates=torch.stack([sample_indices[:,1],sample_indices[:,0],torch.ones_like(sample_indices[:,0])],dim=1)           
            coordinates=coordinates.type(torch.float)
            x=coordinates[:,0]
            projection=torch.matmul(T_matrix,torch.transpose(coordinates,0,1))
            x_prime=projection[0,:]
            y_prime=projection[1,:]
            y_square=torch.pow(y_prime,2)
            y_const=torch.ones_like(y_prime)
            y=torch.stack([y_square,y_prime,y_const],dim=1)
            coefficients=torch.matmul(torch.inverse(torch.matmul(torch.transpose(y,0,1),y)),torch.matmul(torch.transpose(y,0,1),x_prime))            
            x_prime_predict=torch.matmul(y,coefficients)
            matrix_prime_predict=torch.stack([x_prime_predict,y_prime,torch.ones_like(y_prime)],dim=1)
            matrix_predict=torch.matmul(torch.inverse(T_matrix),torch.transpose(matrix_prime_predict,0,1))
            loss+=torch.mean(torch.pow(x-matrix_predict[0,:],2))
    return loss/batch


if __name__=='__main__':  
    EPOCH=50

    train_sampler=RandomSampler(HnetData('./data'))
    train_loader=DataLoader(HnetData('./data'),batch_size=10,sampler=train_sampler,drop_last=True) 
    model=HNet()

    device=torch.device('cuda',0)
    model=model.to(device)

    model=model.train()
    params=model.parameters()
    optimizer=torch.optim.Adam(params,lr=3e-2)

    start_time=int(time.time())
    log=open('./logs/loggings/LaneNet_{}.txt'.format(start_time),'w')
    step=0
    for e_p in range(EPOCH):
        for batch_data in train_loader:
            s=time.time()
            lane_images=batch_data[0]
            instance_label=batch_data[1]

            lane_images=lane_images.cuda()
            instance_label=instance_label.cuda()
            #f_time=time.time()
            outputs=model(lane_images)
            #print("forward time:{}\n".format(time.time()-f_time))
            #l_time=time.time()
            total_loss=hnet_loss(10,outputs,instance_label)
            #print("loss time:{}\n".format(time.time()-l_time))
            log.write('Steps:{}, Total Loss:{}\n'.format(step,total_loss))
            log.flush()
            #g_time=time.time()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            #print("gradient compute time:{}\n".format(time.time()-g_time))
            step+=1
            e=time.time()
            print("step time:{}, total_loss:{:.6f}\n".format(e-s,total_loss))
        torch.save(model.state_dict(),os.path.join('./logs/models','model_1_{}_{}.pkl'.format(start_time,e_p)))
    log.close()

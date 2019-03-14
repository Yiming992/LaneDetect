import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader,SubsetRandomSampler
from Data import TusimpleData
from model import LaneNet
import time
import os
import cv2
import random
from torch.nn.utils import clip_grad_value_
from torch.nn import DataParallel
from loss import Losses

#函数用于将Pytorch数据集分割为训练集与测试集
def split_dataset(test_ratio=0.2):
    dataset_size=len(os.listdir(os.path.join('./data','LaneImages')))
    indices=list(range(dataset_size))
    split=round(dataset_size*test_ratio)
    random.shuffle(indices)
    train_indices=indices[split:]
    test_indices=indices[:split]
    return train_indices,test_indices

#函数用于构建数据采样器
def build_sampler(data,train_batch_size,test_batch_size,train_index,test_index):
    train_sampler=SubsetRandomSampler(train_index)
    test_sampler=SubsetRandomSampler(test_index)
    train_loader=DataLoader(data,batch_size=train_batch_size,sampler=train_sampler,drop_last=True)
    test_loader=DataLoader(data,batch_size=test_batch_size,sampler=test_sampler,drop_last=True)
    return {'train':train_loader,'test':test_loader}

#################################################
class Train:
    def __init__(self,model,data,epoch,batch_size,loss,loss_params,ops_params,
                 lr=5e-4,optimizer='Adam',mode='Parallel',
                 continue_train=False,save=None):
        '''
        用于模型训练的模块
        变量：
           model:网络模型定义
           data:pytorch datasampler物体
           epoch:训练主周期
           batch_size:每批数据量
           loss:损失函数定义
           loss_params:损失函数参数
           ops_params:优化器参数
           lr:学习率
           optimizer:优化器种类
           mode:训练模式
           continue_train:本次训练是否承接已有训练
           save:已训练模型的存贮位置
        '''
        self.model=model
        self.data=data
        self.epoch=epoch
        self.batch_size=batch_size 
        self.loss=loss
        self.loss_params=loss_params
        self.ops_params=ops_params
        self.lr=lr
        self.optimizer=optimizer
        self.mode=mode
        self.continue_train=continue_train
        self.save=save 
                        
    def _train(self):
        if self.mode=='gpu':
            device=torch.device('cuda',0)
            if self.continue_train==True:
                self.model.load_state_dict(torch.load(self.save))
            self.model=self.model.to(device)
        elif self.mode=='parallel':
            num_gpu=torch.cuda.device_count()
            self.model=DataParallel(self.model,device_ids=[i for i in range(num_gpu)])
            if self.continue_train==True:
                self.model.load_state_dict(torch.load(self.save))
            self.model=self.model.cuda()
        self.model.train()
        params=self.model.parameters()
        optimizer=self._create_optimizer()
        optimizer=optimizer(params,lr=self.lr,**self.ops_params)

        start_time=int(time.time())
        log=open('./logs/loggings/LaneNet_{}.txt'.format(start_time),'w')
        step=0
        for e_p in range(self.epoch):
            for batch_data in self.data['train']:
                s=time.time()
                input_data=batch_data[0]
                seg_mask=batch_data[1]
                instance_mask=batch_data[2]

                input_data=input_data.cuda()
                seg_mask=seg_mask.cuda()
                instance_mask=instance_mask.cuda()

                predictions,embeddings=self.model(input_data)
                total_loss=self.loss(input_data,self.batch_size,predictions,
                                     seg_mask,embeddings,instance_mask,**self.loss_params)

                log.write('Steps:{}, Loss:{}\n'.format(step,total_loss))
                log.flush()

                optimizer.zero_grad()
                total_loss.backward()
                clip_grad_value_(model.parameters(),clip_value=5.)
                optimizer.step()
                step+=1
                e=time.time
                print("step time:{}".format(e-s))
            torch.save(model.state_dict(),os.path.join('./logs/models','model_1__{}_{}.pkl'.format(start_time,e_p)))
        log.close()

    def _create_optimizer(self):
        if self.optimizer=='adam':
            return torch.optim.Adam
        elif self.optimizer=='sgd':
            return torch.optim.SGD

    def _lr_adjust(self):
        pass

    def __call__(self):
        self._train()

if __name__=='__main__':
    ap=argparse.ArgumentParser() 
 
    ap.add_argument('-e','--epoch',default=2)#Epoch
    ap.add_argument('-b','--batch',default=8)#Batch_size
    ap.add_argument('-dv','--delta_v',default=.5)#delta_v
    ap.add_argument('-dd','--delta_d',default=6)#delta_d
    ap.add_argument('-l','--learning_rate',default=5e-4)#learning_rate
    ap.add_argument('-o','--optimizer',default='adam')#optimizer
    ap.add_argument('-m','--mode',default='gpu')#training mode,single GPU or multi GPU in parallel
    ap.add_argument('-t','--test_ratio',default=.01)#percent of data to be used in testing
    ap.add_argument('-ct','--continue_train',default='no')#whether the current training loop is a continuation of a previous one
    ap.add_argument('-s','--save',default=None)#location of the saved model checkpoint file

    args=vars(ap.parse_args())
    train_indices,test_indices=split_dataset(args['test_ratio'])
    data=build_sampler(TusimpleData('./data'),args['batch'],1,train_indices,test_indices)    
    model=LaneNet()
    loss=Losses
    loss_parameters={'delta_v':args['delta_v'],'delta_d':args['delta_d'],'alpha':1,'beta':1,'gamma':.001}
    optimizer_parameters={'amsgrad':True}

    if args['continue_train']=='Yes':
        train=Train(model,data,args['epoch'],args['batch'],
                    loss,loss_parameters,args['learning_rate'],optimizer_parameters
                    optimizer=args['optimizer'],mode=args['device'],continue_train=True,
                    save=args['save'])
        train()
    else:
        train=Train(model,data,args['epoch'],args['batch'],
                    args['delta_v'],args['delta_d'],args['learning_rate'],
                    optimizer=args['optimizer'],mode=args['device'])
        train()
    
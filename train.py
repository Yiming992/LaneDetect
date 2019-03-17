import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader,SubsetRandomSampler
from Data import TusimpleData,Rescale
from model1 import LaneNet
from loss import Segmentation_loss,instance_loss,bi_weight
import time
import os
import cv2
import random
from torch.nn.utils import clip_grad_value_
from torch.nn import DataParallel


def split_dataset(test_ratio=0.2):
    dataset_size=len(os.listdir(os.path.join('./data','LaneImages')))
    indices=list(range(dataset_size))
    split=round(dataset_size*test_ratio)
    random.shuffle(indices)
    train_indices=indices[split:]
    test_indices=indices[:split]
    return train_indices,test_indices

def build_sampler(data,train_batch_size,test_batch_size,train_index,test_index):
    train_sampler=SubsetRandomSampler(train_index)
    test_sampler=SubsetRandomSampler(test_index)
    train_loader=DataLoader(data,batch_size=train_batch_size,sampler=train_sampler,drop_last=True)
    test_loader=DataLoader(data,batch_size=test_batch_size,sampler=test_sampler,drop_last=True)
    return {'train':train_loader,'test':test_loader}

def compute_loss(predictions,embeddings,seg_mask,instance_mask,
                 class_weight,delta_v,delta_d):
    seg_loss=Segmentation_loss(predictions,seg_mask,class_weight)
    total,Variance,Distance,Reg=instance_loss(delta_v,delta_d,embeddings,instance_mask)
    total_loss=seg_loss+total
    return total_loss,Variance,Distance,Reg

def train(model,data,epoch,batch,delta_v,
          delta_d,lr=3e-5,optimizer='Adam',mode='GPU',
          continue_train=False,save=None):
    if mode=='GPU':
        device=torch.device('cuda',0)
        if continue_train==True:
            model.load_state_dict(torch.load(save))
        model=model.to(device)
    elif mode=='Parallel':
        num_gpu=torch.cuda.device_count()
        model=DataParallel(model,device_ids=[i for i in range(num_gpu)])
        if continue_train==True:
            model.load_state_dict(torch.load(save))
        model=model.cuda()
    else:
        device=torch.device('cpu',0)
        if continue_train==True:
            model.load_state_dict(torch.load(save))
        model=model.to(device)
    model.train()
    params=model.parameters()
    optimizer=torch.optim.Adam(params,lr=lr)
    start_time=int(time.time())
    log=open('./logs/loggings/LaneNet_{}.txt'.format(start_time),'w')
    step=0
    for e_p in range(epoch):
        for batch_data in data['train']:
            s=time.time()
            input_data=batch_data[0]
            seg_mask=batch_data[1]
            instance_mask=batch_data[2]
           
            
            class_weight=bi_weight(seg_mask,batch)          
            input_data=input_data.cuda()
            seg_mask=seg_mask.cuda()
            instance_mask=instance_mask.cuda()
            predictions,embeddings=model(input_data)
            total_loss,Variance,Distance,Reg=compute_loss(predictions,embeddings,seg_mask,instance_mask,
                                                          class_weight,delta_v,delta_d)                              
            log.write('Steps:{}, Loss:{}\n'.format(step,total_loss))
            log.flush()
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_value_(model.parameters(),clip_value=5.)
            optimizer.step()
            step+=1
            e=time.time()
            print('step time:{}'.format(e-s))     
        torch.save(model.state_dict(),os.path.join('./logs/models','model_1_{}_{}.pkl'.format(start_time,e_p)))
    log.close()
            
if __name__=='__main__':
    ap=argparse.ArgumentParser() 
 
    ap.add_argument('-e','--epoch',default=300)
    ap.add_argument('-b','--batch',default=8)
    ap.add_argument('-dv','--delta_v',default=.5)
    ap.add_argument('-dd','--delta_d',default=6)
    ap.add_argument('-l','--learning_rate',default=5e-4)
    ap.add_argument('-o','--optimizer',default='Adam')
    ap.add_argument('-d','--device',default='GPU')
    ap.add_argument('-t','--test_ratio',default=.01)
    ap.add_argument('-ct','--continue_train',default='No')
    ap.add_argument('-s','--save',default=None)
   
    args=vars(ap.parse_args())
    train_indices,test_indices=split_dataset(args['test_ratio'])
    data=build_sampler(TusimpleData('./data',transform=Rescale((256,512))),args['batch'],1,train_indices,test_indices)    
    model=LaneNet()

    if args['continue_train']=='Yes':
        train(model,data,args['epoch'],args['batch'],
              args['delta_v'],args['delta_d'],args['learning_rate'],
              optimizer=args['optimizer'],mode=args['device'],continue_train=True,
              save=args['save'])
    else:
        train(model,data,args['epoch'],args['batch'],
              args['delta_v'],args['delta_d'],args['learning_rate'],
              optimizer=args['optimizer'],mode=args['device'])
     







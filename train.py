import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader,SubsetRandomSampler
from Data import TusimpleData,Rescale
from model import LaneNet
from HNet import HNet
from loss import Segmentation_loss,instance_loss,bi_weight
import time
import os
import cv2
import random
from torch.nn.utils import clip_grad_value_

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
          delta_d,lr=3e-5,optimizer='Adam',mode='GPU'):
    if mode=='GPU':
        device=torch.device('cuda',0)
    else:
        device=torch.device('cpu',0)
    model.to(device)
    model.train()
    params=model.parameters()
    optimizer=torch.optim.Adam(params,lr=lr)
    start_time=int(time.time())
    log=open('./logs/loggings/LaneNet_{}.txt'.format(start_time),'w')
    for e_p in range(epoch):
        for batch_id,batch_data in enumerate(data['train']):
            input_data=batch_data[0]
            seg_mask=batch_data[1]
            instance_mask=batch_data[2]

            class_weight=bi_weight(seg_mask,batch)          
            input_data=input_data.to(device)
            seg_mask=seg_mask.to(device)
            instance_mask=instance_mask.to(device)
            predictions,embeddings=model(input_data)
            total_loss,Variance,Distance,Reg=compute_loss(predictions,embeddings,seg_mask,instance_mask,
                                    class_weight,delta_v,delta_d)                               
            log.write('Steps:{}, Loss:{}\n'.format(batch_id*(e_p+1),total_loss))
            log.flush()
            #print('v:{},d:{},r:{}'.format(Variance,Distance,Reg))
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_value_(model.parameters(),clip_value=5.)
            optimizer.step()
            #print(list(model.parameters())[0])
            #print(list(model.parameters())[0].grad)
        
        torch.save(model,os.path.join('./logs/models','model_{}_{}.pkl'.format(start_time,e_p)))
    log.close()

            
if __name__=='__main__':
    ap=argparse.ArgumentParser() 
 
    ap.add_argument('-e','--epoch',default=50)#Epoch
    ap.add_argument('-b','--batch',default=8)#Batch_size
    ap.add_argument('-dv','--delta_v',default=.5)#delta_v
    ap.add_argument('-dd','--delta_d',default=3)#delta_d
    ap.add_argument('-l','--learning_rate',default=5e-4)#learning_rate
    ap.add_argument('-o','--optimizer',default='Adam')#optimizer
    ap.add_argument('-d','--device',default='GPU')#training device
    ap.add_argument('-t','--test_ratio',default=.1)
    ap.add_argument('-s','--stage',default='new')
    #ap.add_argument('-cl','--class_weight',default=.5)
    #ap.add_argument()
    #ap.add_argument()
    
    args=vars(ap.parse_args())
    train_indices,test_indices=split_dataset(args['test_ratio'])
    data=build_sampler(TusimpleData('./data',transform=Rescale((256,512))),args['batch'],1,train_indices,test_indices)
    
    if args['stage']=='new':  
        model=LaneNet()
    else:
        model_file='model_1548830895.pkl'
        model=LaneNet()
        weight_dict=torch.load(os.path.join('./logs/models/',model_file))
        model.load_state_dict(weight_dict.state_dict())
 
    train(model,data,args['epoch'],args['batch'],
          args['delta_v'],args['delta_d'],args['learning_rate'],
          args['optimizer'])
     







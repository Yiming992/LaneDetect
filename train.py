import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader,SubsetRandomSampler
from Data import TusimpleData,Rescale
from model import LaneNet
from HNet import HNet
from loss import Segmentation_loss,variance,distance
import datetime
from torchvision import transforms
import os
import random

#Transform=transforms.Compose([Rescale((512,256))])

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
    train_loader=DataLoader(data,batch_size=train_batch_size,sampler=train_sampler)
    test_loader=DataLoader(data,batch_size=test_batch_size,sampler=test_sampler)
    return train_loader,test_loader

def compute_loss(predictions,embeddings,seg_mask,instance_mask,
                 class_weight,delta_v,delta_d):
    seg_loss=Segmentation_loss(predictions,seg_mask,class_weight)
    variance=variance(delta_v,embeddings,instance_mask)
    distance=distance(delta_b,embeddings,instance_mask)
    total_loss=seg_loss+.5*variance+.5*distance
    return total_loss


def train(model,data,epoch,batch,class_weight,deelta_v,
          delta_d,lr=3e-5,optimizer='Adam',mode='GPU',):
    if mode=='GPU':
        device=torch.device('cuda')
    model.to(device)
    model.train()
    params=model.parameters()
    optimizer=torch.optim.Adam(params,lr=lr)
    start_time=datetime.datetime.now()
    log=open('./logs/loggings/LaneNet_{}.txt'.format(start_time),'w')
    for e_p in range(epoch):
        for batch_id,batch_data in enumerate(data['train']):
            input_data=batch_data[0]
            seg_mask=batch_data[1]
            instance_mask=batch_data[2]          
            input_data=input_data.to(device)
            seg_mask=seg_mask.to(device)
            instance_mask=instance_mask.to(device)
            predictions,embeddings=model(input_data)
            total_loss=compute_loss(predictions,embeddings,mask,seg_mask,instance_mask,
                                    class_weight,delta_v,delta_d)         
            log.write('Steps:{},Loss:{}'.format(batch_id*(e_p+1),total_loss))
            log.flush()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    torch.save(model,os.path.join('./logs/models','model_{}.pkl'.format(start_time)))
    log.close()

            


if __name__=='__main__':
    ap=argparse.ArgumentParser() 
 
    ap.add_argument('-e','--epoch',required=True,default=10)#Epoch
    ap.add_argument('-b','--batch',required=True,default=4)#Batch_size
    ap.add_argument('-dv','--delta_v',required=True,default=1)#delta_v
    ap.add_argument('-dd','--delta_d',required=True,default=1)#delta_d
    ap.add_argument('-l','--learning_rate',required=True,default=3e-5)#learning_rate
    ap.add_argument('-o','--optimizer',required=True,default='Adam')#optimizer
    ap.add_argument('-d','--device',required=True,default='GPU')#training device
    ap.add_argument('t','--test_ratio',required=True,default=.2)
    ap.add_argument('cl','class_weight',required=True,default=.5)
    #ap.add_argument()
    #ap.add_argument()
    #

    args=vars(ap.parse_args())
    
    train_indices,test_indices=split_dataset(args['test_ratio'])
    train_sampler,test_sampler=build_sampler(Tusimple_data,args['batch'],1,
                                             train_indices,test_indices)
    
    model=LaneNet()

    train(model,train_sampler,args['epoch'],args['batch'],args['learning_rate'],
          args['optimizer'],args['device'],args['class_weight'],args['delta_v'],
          args['delta_d'])








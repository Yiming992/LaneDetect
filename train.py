import argparse
import shuffle
import numpy as np 
import torch 
from torch.utils.data import DataLoader,SubsetRandomSampler
from Data import Tusimple_data,Rescale
from model import LaneNet
from HNet import HNet
from loss import Segmentation_loss,variance,distance

Transform=transforms.Compose([Rescale((512,256))])

def split_dataset(test_ratio=0.2):
    dataset_size=len(os.listdir(os.path.join('./data','LaneImages')))
    indices=list(range(dataset_size))
    split=dataset_size*test_ratio
    random.shuffle(indices)
    train_indices=indices[split:]
    test_indices=indices[:split]

    return train_indices,test_indices

def build_sampler(data,train_batch_size,test_batch_size,train_index,test_index):
    train_sampler=SubSetRandomSampler(train_indices)
    test_sampler=SubSetRandomSampler(test_indices)

    train_loader=DataLoader(data,batch_size=batch_size,sampler=train_sampler)
    test_loader=DataLoader(data,batch_size=batch_size,sampler=test_sampler)

    return train_loader,test_loader

##data={'input_data','binary_mask','instance_mask'}
def train_monitor(func):
    # *args: epoch, batch,
    def wrapper(*args,**kwargs):
        for epoch in range(epochs):
            if mode=='GPU':
                device=torch.device('cuda' if torch.cuda.is_available())
                LR=lr
                if optimizer=='Adam':
                    optimizer=torch.optim.Adam()           
            num_batches=train_data 
            for batch in   
    return wrapper


def train(model,data,epoch,batch,lr=3e-5,optimizer='Adam',mode='GPU'):
    if mode=='GPU':
        device=torch.device('cuda')
    optimizer=torch.optim.Adam(lr=lr)

    model.to(device)
    model.train()
    for e_p in range(epoch):
        for batch_id,batch_data in enumerate(data['train']):
            input_data=batch_data[0]
            seg_mask=batch_data[1]
            instance_mask=batch_data[2]
            
            
            input_data=input_data.to(device)
            seg_mask=seg_mask.to(device)
            instance_mask=instance_mask.to(device)
            
            

 

def compute_loss(predictions,embeddings,mask,labels,
                 class_weight,delta_v,delta_d):

    seg_loss=Segmentation_loss(predictions,label,class_weight)
    variance=variance(delta_v,embeddings,labels)
    distance=distance(delta_b,embeddings,labels)

    total_loss=seg_loss+.5*variance+.5*distance

    return total_loss

if __name__=='__main__':
    ap=argparse.ArgumentParser() 
 
    ap.add_argument('-e','--epoch',required=True,default=10)#Epoch
    ap.add_argument('-b','--batch',required=True,default=4)#Batch_size
    ap.add_argument('-dv','--delta_v',required=True,default=1)#delta_v
    ap.add_argument('-dd','--delta_d',required=True,default=1)#delta_d
    ap.add_argument('-l','--learning_rate',required=True,default=3e-5)#learning_rate
    ap.add_argument('-o','--optimizer',required=True,default='Adam')#optimizer
    #

    args=vars(ap.parse_args())

    
    LaneNet=LaneNet()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data[]
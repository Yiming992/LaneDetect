import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader,SubsetRandomSampler
from Data import Tusimple_data,Rescale
from model import LaneNet
from HNet import HNet
from loss import Segmentation_loss,Clustering_loss,Hnet_loss

Transform=transforms.Compose([Rescale()])
sampler=SubsetRandomSampler()

dataset_size=len(os.listdir(os.path.join('./data','LaneImages')))
indices=list(range(dataset))
split=dataset_size*.8
np.random.shuffle(indices)
train_indices,test_indices=indices[:split],indices[split:]

train_sampler=SubSetRandomSampler(train_indices)
test_sampler=SubSetRandomSampler(test_indices)

train_loader=DataLoader(Tusimple_data,sampler=train_sampler)
test_loader=DataLoader(Tusimple_data,sampler=test_sampler)


def dataset_subset()

##data={'input_data','binary_mask','instance_mask'}
def train_monitor(func,data,epoch,batch,
                  lr=3e-5,optimizer='Adam',device='GPU'):
    def wrapper():
        for epoch in range(epochs):
            if mode=='GPU':
                device=torch.device('cuda' if torch.cuda.is_available())
                LR=lr
                if optimizer=='Adam':
                    optimizer=torch.optim.Adam() 
            for batch
    return wrapper

@train_monitor
def train(model,epoch,batch,lr,optimizer,device):
    pass


if __name__=='__main__':
    ap=argparse.ArgumentParser()
 
    ap.add_argument('-e','--epoch',required=True,default=10)#Epoch
    ap.add_argument('-b','--batch',required=True,default=4)#Batch_size
    ap.add_argument('-dv','--delta_v',required=True,default=1)#delta_v
    ap.add_argument('-dd','--delta_d',required=True,default=1)#delta_d
    ap.add_argument('-l','--learning_rate',required=True,default=3e-5)#learning_rate
    ap.add_argument('-o','--optimizer',required=True,default='Adam')#optimizer
    ap.add_argument('-d','--device',required=True,default='GPU')
    #

    args=vars(ap.parse_args())


    LaneNet=LaneNet()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



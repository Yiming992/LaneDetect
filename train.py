import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from Data import Tusimple_data,Rescale
from model import LaneNet
from HNet import HNet
from loss import Segmentation_loss,Clustering_loss,Hnet_loss



def train_monitor(func,data,epochs,batch,lr=3e-5,optimizer='Adam',mode='GPU'):

    def wrapper():
        for epoch in range(epochs):
    pass

@train_monitor
def train(model,epoch,batch_size,lr,optimizer,device):
    pass


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
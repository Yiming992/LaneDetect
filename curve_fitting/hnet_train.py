import argparse
import numpy as np 
import torch 
from HNet import HNet,hnet_loss
from hnet_data import Hnet_Data
import time
from torch.nn.utils import clip_grad_value_
import os

def train(model,image_path,lane_path,epoch,lr=3e-5,optimizer='Adam',mode='GPU'):
    if mode=='GPU':
        device=torch.device('cuda',0)
    else:
        device=torch.device('cpu',0)
    model.to(device)
    model.train()
    params=model.parameters()
    optimizer=torch.optim.Adam(params,lr=lr)
    start_time=int(time.time())
    log=open('./logs/loggings/Hnet_{}.txt'.format(start_time),'w')
    data_loader=Hnet_Data(image_path,lane_path)
    step=0
    for e_p in range(epoch):
        for data in data_loader.fetch():
            step+=1           
            log.write('Steps:{}, Loss:{}\n'.format(step,total_loss))
            log.flush()
            image=data['image']
            image=image.cuda()
            coefficient=model(image)
            total_loss=hnet_loss(coefficient,data)
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_value_(model.parameters(),clip_value=5.)
            optimizer.step()
        
        torch.save(model.state_dict(),os.path.join('./logs/models','hnet_{}_{}.pkl'.format(start_time,e_p)))
    log.close()

            
if __name__=='__main__':
    ap=argparse.ArgumentParser() 
 
    ap.add_argument('-e','--epoch',default=50)#Epoch
    ap.add_argument('-i','--image_path',default='./data/LaneImages')#delta_v
    ap.add_argument('-la','--lane_path',default='./data/cluster')#delta_d
    ap.add_argument('-l','--learning_rate',default=5e-4)#learning_rate
    ap.add_argument('-o','--optimizer',default='Adam')#optimizer
    ap.add_argument('-d','--device',default='GPU')#training device
    ap.add_argument('-s','--stage',default='new')
    #ap.add_argument('-cl','--class_weight',default=.5)
    #ap.add_argument()
    #ap.add_argument() 
    args=vars(ap.parse_args())
    
    if args['stage']=='new':  
        model=HNet()
    else:
        model_file='model_1548830895.pkl'
        model=HNet()
        weight_dict=torch.load(os.path.join('./logs/models/',model_file))
        model.load_state_dict(weight_dict.state_dict())
 
    train(model,args['image_path'],args['lane_path'],args['epoch'],args['learning_rate'],
          optimizer=args['optimizer'],mode=args['device'])

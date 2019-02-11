from clustering import lane_cluster 
from model import LaneNet
import torch
import numpy as np
import cv2
import argparse
### To be completed
if __name__=='__main__':

    ap=argparse.ArgumentParser()
    
    ap.add_argument('-m','--mode',default='single',required=True)
    ap.add_argument('-a','--address',required=True)
    ap.add_argument('-d','--device',default='GPU')
    ap.add_argument('-s','--save',default='./logs/predictions')

    args=vars(ap.parse_args())

    if args['device']=='GPU':
        device=torch.device('cuda',0)
    else:
        device=torch.device('cpu')
    
    model=LaneNet()
    model=model.to(device)
    model.eval()
    if args['mode']=='single':
        img=cv2.imread(args['address'],cv2.IMREAD_UNCHANGED)
        img=cv2.resize(img,(256,512))
        img=img/255
        img=torch.tensor(img)
        img=img.to(device)
        prediction=model(img)
        prediction=prediction.data.cpu().numpy()
        cv2.imwrite(args['save'])
    elif args['mode']=='batch':
        img_files=os.listdir(args['address'])


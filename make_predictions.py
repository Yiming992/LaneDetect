<<<<<<< HEAD
from clustering import lane_cluster 
=======
from clustering import cluster 
>>>>>>> d36b553c9bf311ed1bff2cc988c4338a6af43c12
from model import LaneNet
import torch
import numpy as np
import cv2
<<<<<<< HEAD
import argparse
from torch.nn import DataParallel
import os

def forward(model,img_path):
    img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(256,512))
    image=img
    img=np.transpose(img,axes=[2,0,1])
    img=torch.tensor(img,dtype=torch.float)
    img=img.unsqueeze(dim=0)
    img=img.cuda()  
    segmentation,embedding=model(img)
    return image,binary_mask,embedding

def gen_binary(segmentation,threshold=.5):
    segmentation=segmentation.data.cpu().numpy()
    binary_mask=segmentation.squeeze()
    binary_mask=np.exp(binary_mask)
    binary_mask=binary_mask/binary_mask.sum(axis=0)
    if threshold:
        binary_mask=binary_mask[1,:,:]>threshold
        binary_mask=binary_mask.astype(np.float)
    return binary_mask
    
def gen_instance(image,binary_mask,embedding):
    clustering=lane_cluster(None,image,binary_mask,embedding)
    instance_mask,labels=clustering()
    return instance_mask

if __name__=='__main__':

    ap=argparse.ArgumentParser()
    
    ap.add_argument('-m','--mode',default='single',required=True)
    ap.add_argument('-a','--address',required=True)
    ap.add_argument('-d','--device',default='GPU')
    ap.add_argument('-s','--save',default='./logs/predictions')
    ap.add_argument('-p','--parallel',default='Yes')
    ap.add_argument('-w','--weights',required=True)

    args=vars(ap.parse_args())

    model=LaneNet()

    if args['device']=='GPU':
        if args['parallel']=='Yes':
            model=DataParallel(model)
        model.load_state_dict(args['weights'])
        model.cuda()
        model.eval()
    else:
        model.load_state_dict(args['weights'])
        model.eval()
    
    if args['mode']=='single':
        image,segmentation,embedding=forward(model,args['address'])
        binary_mask=gen_binary(segmentation)
        instance_mask=gen_instance(image,binary_mask,embedding)

        cv2.imwrite(args['save'],instance_mask)
    elif args['mode']=='batch':
        img_files=os.listdir(args['address'])
        for i in img_files:
            image,segmentation,embedding=forward(model,args['address'])
            binary_mask=gen_binary(segmentation)
            instance_mask=gen_instance(image,binary_mask,embedding)
            cv2.imwrite(os.path.join(args['save'],i),instance_mask)




=======
import os

### To be completed
if __name__=='__main__':

    saved_model=os.listdir('./logs/models')
    model=LaneNet()
    =torch.load()









>>>>>>> d36b553c9bf311ed1bff2cc988c4338a6af43c12

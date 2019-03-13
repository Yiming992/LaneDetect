import os 
import numpy as np 
import cv2
from model1 import LaneNet 
from torch.nn import DataParallel
from clustering import lane_cluster
import torch

SAVE_PATH='./test_result'
IMAGE_PATH='./train_set/clips/0313-2/220'
MODEL_SAVE='./logs/models/model_1_1552383528_299.pkl'

if __name__=='__main__':

    model=LaneNet()
    model=DataParallel(model)
    model.load_state_dict(torch.load(MODEL_SAVE))
    model=model.cuda()
    model.eval()

    image_files=os.listdir(IMAGE_PATH)
    for i in image_files:
        image=cv2.imread(os.path.join(IMAGE_PATH,i),cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(256,512))
        img=image


        image=image.transpose(2,0,1)
        image=image[np.newaxis,:,:,:]
        image=image/255
        image=torch.tensor(image,dtype=torch.float)

        segmentation,embedding=model(image)

        binary_mask=segmentation.data.cpu().numpy()
        binary_mask=binary_mask.squeeze()

        exp_binary_mask=np.exp(binary_mask)
        exp_binary_mask=exp_binary_mask.sum(axis=0)
        binary_mask=np.exp(binary_mask)/exp_binary_mask
        
        if not os.path.exists('./test_result/binary'):
            os.mkdir('./test_result/binary')
        cv2.imwrite(os.path.join('./test_result/binary',i),binary_mask[1,:,:]*255)

        threshold_mask=binary_mask[1,:,:]>.5
        threshold_mask=threshold_mask.astype(np.float)

        cluster=lane_cluster(1.5,img,embedding.squeeze().data.cpu().numpy(),threshold_mask,mode='point',method='Meanshift')
        instance_mask=cluster()
        
        if not os.path.exists('./test_result/instance'):
            os.mkdir('./test_result/instance')
        cv2.imwrite(os.path.join('./test_result/instance','.'.join([i.split('.')[0],'png'])),instance_mask)

    

        



    
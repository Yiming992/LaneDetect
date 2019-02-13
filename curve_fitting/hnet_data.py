import os 
import cv2
import torch
import numpy as np
import sys 
from Data import clean_values
from collections import defaultdict

sys.path.append('../')

class Hnet_Data:
    def __init__(self,image_path,lane_path,ranodm_permute=True):
        self.img_path=image_path
        self.lane_path=lane_path
        self.permute=random_permute
    
    @staticmethod
    def _imread(image_path):
        image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(256,512))
        return image    
    
    def fetch(self):
        #output:dict{image.tensor,number of lanes,list of coordinate tuple}
        file_list=os.listdir(self.img_path)
        if self.permute:
            file_list=np.random.permutation(file_list)
        for f in file_list:
            data={}
            image=_imread(os.path.join(self.path,f))
            lane_name=f.split('.')[0]
            lane_image=_imread(os.path.join(self.lane_path,lane_name+'.png'))
            image=image.transpose([2,0,1])
            image=image[np.newaxis,:,:,:]
            image=torch.tensor(image)

            data['image_tensor']=image
            unique_lanes=np.unique(lane_image)
            data['num_lane']=len(unique_lanes)-1
            
            coord={}
            for index,v in enumerate(unique_lanes):
                if v==0:
                    continue
                else:
                    coordinates=np.where(lane_image==v)
                    for i in range(coordinates.shape[1]):
                        coord[((coordinates[0,i],coordinates[1,i]))
                    



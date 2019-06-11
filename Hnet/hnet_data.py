from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch
import numpy as np
from torch.utils.data import RandomSampler

##根据已有数据创建Pytorch Dataset
class HnetData(Dataset):
    def __init__(self,root_dir,rescale=(128,64)):
        super(HnetData,self).__init__()
        self.root_dir=root_dir
        self.rescale=rescale
        file_names=os.listdir(os.path.join(self.root_dir,'LaneImages'))
        name_map={}
        for idx,file_name in enumerate(file_names):
            name_map[idx]=file_name.split('.')[0]
        self.name_map=name_map

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,'LaneImages')))
    
    def __getitem__(self,index):
        lane_image=cv2.imread(os.path.join(self.root_dir,'LaneImages',self.name_map[index]+'.jpg'),cv2.IMREAD_UNCHANGED)
        lane_image=cv2.cvtColor(lane_image,cv2.COLOR_BGR2RGB)
        cluster=cv2.imread(os.path.join(self.root_dir,'cluster',self.name_map[index]+'.png'),cv2.IMREAD_UNCHANGED)        
        if self.rescale:
            lane_image=cv2.resize(lane_image,self.rescale,cv2.INTER_CUBIC)       
        lane_image=np.transpose(lane_image,(2,0,1))
        lane_image=torch.tensor(lane_image,dtype=torch.float)/255
        instance_label=torch.tensor(cluster,dtype=torch.float)
        return lane_image,instance_label




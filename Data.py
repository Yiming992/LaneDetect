from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch
import numpy as np

########
def clean_values(sample,target='binary'):
    H,W=sample.shape
    if target=='binary':
        values=[0,255]
        for h in range(H):
            for w in range(W):
                if sample[h,w] not in values:
                    sample[h,w]=0
    else:
        values=[255,205,155,105,55]
        for h in range(H):
            for w in range(W):
                if sample[h,w] not in values:
                    sample[h,w]=0
    return sample

class Rescale():
    def __init__(self,output_size,method='INTER_AREA'):
        self.size=output_size
    
    def __call__(self,sample,target='binary'):
        return cv2.resize(sample,self.size,interpolation=cv2.INTER_AREA)


class TusimpleData(Dataset):
    def __init__(self,root_dir,transform=None):
        super(TusimpleData,self).__init__()
        self.root_dir=root_dir
        self.transform=transform
        file_names=os.listdir(os.path.join(self.root_dir,'LaneImages'))
        name_map={}
        for idx,file_name in enumerate(file_names):
            name_map[idx]=file_name.split('.')[0]
        self.name_map=name_map

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,'LaneImages')))
    
    def __getitem__(self,index):
        lane_image=cv2.imread(os.path.join(self.root_dir,'LaneImages',self.name_map[index]+'.jpg'),cv2.IMREAD_UNCHANGED)
        binary_label=cv2.imread(os.path.join(self.root_dir,'train_binary',self.name_map[index]+'.png'),cv2.IMREAD_UNCHANGED)
        instance_label=cv2.imread(os.path.join(self.root_dir,'cluster',self.name_map[index]+'.png'),cv2.IMREAD_UNCHANGED)
        lane_image=cv2.cvtColor(lane_image,cv2.COLOR_BGR2RGB)

        if self.transform:
            lane_image=self.transform(lane_image)
            binary_label=self.transform(binary_label)
            #binary_label=clean_values(binary_label)
            instance_label=self.transform(instance_label)
            #instance_label=clean_values(instance_label,target='instance')
        
        binary_label=binary_label/255
        lane_image=np.transpose(lane_image,(2,0,1))

        lane_image=torch.tensor(lane_image,dtype=torch.float)/255.
        instance_label=torch.tensor(instance_label,dtype=torch.float)
        return lane_image,binary_label,instance_label

'''
For future new datasets
'''
class NewData(Dataset):
    pass








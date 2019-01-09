from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch 

class Rescale():
    def __init__(self,output_size,method='INTER_AREA'):
        self.size=output_size

    def __call__(self,sample):
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
        
        if self.transform:
            lane_image=self.transform(lane_image)
            binary_label=self.transform(binary_label)
            instance_label=self.transform(instance_label)

        lane_image=torch.tensor(lane_image,dtype=torch.float)
        binary_label=torch.tensor(binary_label,dtype=torch.float)
        instance_label=torch.tensor(instance_label,dtype=torch.float)

        return lane_image,binary_label,instance_label
 
class NewData(Dataset):
    pass










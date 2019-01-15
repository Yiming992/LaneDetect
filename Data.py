from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch
import numpy as np 

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
        lane_image=cv2.cvtColor(lane_image,cv2.COLOR_BGR2RGB)

        if self.transform:
            lane_image=self.transform(lane_image)
            binary_label=self.transform(binary_label)
            instance_label=self.transform(instance_label)
        
        binary_label=np.stack([binary_label/255.,np.ones_like(binary_label)-binary_label/255.],axis=0)
        instance_label=np.stack([instance_label,np.ones_like(instance_label)-instance_label],axis=0)
        lane_image=np.transpose(lane_image,(2,0,1))

        lane_image=torch.tensor(lane_image,dtype=torch.float)/255.
        binary_label=torch.tensor(binary_label,dtype=torch.float)
        instance_label=torch.tensor(instance_label,dtype=torch.float)
        return lane_image,binary_label,instance_label
 
class NewData(Dataset):
    pass

if __name__=='__main__':

    from train import split_dataset,build_sampler
    

    train_indices,test_indices=split_dataset(.2)
    data=build_sampler(TusimpleData('./data',transform=Rescale((256,512))),
                                             4,1,
                                             train_indices,test_indices)
    for i in data['train']:

        print(i[0].size())
        print(i[1].size())







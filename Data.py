from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2 

class Rescale():

    def __init__(self,output_size,method='CV_INTER_AREA'):
        self.size=output_size
        self.method=method

    def __call__(self,sample):
        return cv2.resize(sample,self.size,interpolation=self.method)

class Tusimple_data(Dataset):

    def __init__(self,root_dir,transform=None):
        super(Tusimple_data).__init__()
        self.root_dir=root_dir
        self.transform=transform
        file_names=os.listdir(os.path.join(self.root_dir))
        name_map={}
        for idx,file_name in enumerate(file_names):
            name_map[idx]=file_name.split('.')[0]
        self.name_map=name_map

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,'LaneImages')))
    
    def __getitem__(self,index):
        lane_image=cv2.imread(os.path.join(self.root_dir,'LaneImages',name_map+'.jpg'))
        binary_label=cv2.imread(os.path.join(self.root_dir,'train_binary',name_map+'.png'))
        instance_label=cv2.imread(os.path.join(self.root_dir,'cluster',name_map+'.png'))

        if self.transform:
            lane_image=self.transform(lane_image)
            binary_label=self.transform(binary_label)
            instance_label=self.transform(instance_label)

        return lane_image,binary_label,instance_label
 
class New_Data(Dataset):
    pass










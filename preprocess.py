import cv2
import json 
import numpy as np
import os
from copy import deepcopy 

TUSIMPLE_PATH='./train_set'

class Rescale():
    def __init__(self,output_size,method='INTER_AREA'):
        self.size=output_size
    
    def __call__(self,sample,target='binary'):
        sample=cv2.resize(sample,self.size,interpolation=cv2.INTER_NEAREST)
        return sample
 
##create datasets for training from uncompressed tusimple data
class CreateTusimpleData():

    def __init__(self,tusimple,line_width,transform=Rescale((512,256))):

        self.tusimple=tusimple 
        self.line_width=line_width
        self.transform=transform
        
    def __call__(self):
        if not os.path.exists('./data/train_binary'):
            os.mkdir('./data/train_binary')
        if not os.path.exists('./data/cluster'):
            os.mkdir('./data/cluster')
        if not os.path.exists('./data/LaneImages'):
            os.mkdir('./data/LaneImages')
        jsons=[json for json in os.listdir(self.tusimple) if json.split('.')[-1]=='json']
        for j in jsons:
            data=[]
            with open(os.path.join(self.tusimple,j)) as f:
                for line in f.readlines():
                    data.append(json.loads(line))
            for entry in data:
                height=entry['h_samples']
                width=entry['lanes']
                clip=entry['raw_file']
                img=cv2.imread(os.path.join(self.tusimple,clip))
                cv2.imwrite(os.path.join('./data/LaneImages','_'.join(clip.split('/')[1:])),img)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img_binary=np.zeros(img.shape,dtype=np.uint8)
                img_cluster=np.zeros(img.shape,dtype=np.uint8)
                for lane in range(len(width)):
                    coordinate=[]
                    for w,h in zip(width[lane],height):
                        if w==-2:
                            continue
                        else:
                            coordinate.append(np.array([w,h],dtype=np.int32))
                    img_binary=cv2.polylines(img_binary,np.stack([coordinate]),isClosed=False,color=255,thickness=5)
                    img_cluster=cv2.polylines(img_cluster,np.stack([coordinate]),isClosed=False,color=255-lane*50,thickness=5)

                name_list=clip.split('/')[1:]

                new_name='_'.join(name_list)
                new_name='.'.join([new_name.split('.')[0],'png'])

                img_binary=self.transform(img_binary)
                img_cluster=self.transform(img_cluster,target='instance')

                cv2.imwrite(os.path.join('./data/train_binary',new_name),img_binary)
                cv2.imwrite(os.path.join('./data/cluster',new_name),img_cluster)

      
if __name__=='__main__':

    creator=CreateTusimpleData(TUSIMPLE_PATH,16)
    creator()



    






    

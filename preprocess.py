import json
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import cv2
import torch

TUSIMPLE_PATH='../train_set'

##标注并结构化图森数据集
class create_tusimple_data():

    def __init__(self,tusimple,line_width):

        self.tusimple=tusimple 
        self.line_width=line_width

    def __call__(self):
        if not os.path.exists('./data/train_binary'):
            os.mkdir('./data/train_binary')
        if not os.path.exists('./data/cluster'):
            os.mkdir('./data/cluster')
        if not os.path.exists('./data/train'):
            os.mkdir('./data/train')
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
                cv2.imwrite(os.path.join('./data/train',clip),img)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img_binary=np.zeros_like(img)
                img_cluster=np.zeros_like(img)
                for lane in range(len(width)):
                    queue=[]
                    for h,w in zip(height,width[lane]):
                        if w<0:
                            continue
                        else:
                            queue.insert(0,(w,h))
                            if len(queue)==2:
                                cv2.line(img_binary,queue[0],queue[1],255,self.line_width)
                                cv2.line(img_cluster,queue[0],queue[1],255-50*lane,self.line_width)
                            if len(queue)>1:
                                queue.pop()
                new_name='_'.join(clip.split('/')[1:])
                new_name='.'.join([new_name.split('.')[0],'png'])

                cv2.imwrite(os.path.join('./data/train_binary',new_name),img_binary)
                cv2.imwrite(os.path.join('./data/cluster',new_name),img_cluster)


class Rescale():

    def __init__(self,output_size,method):
        self.size=output_size
        self.method=method

    def __call__(self,sample):
        return cv2.resize(sample,self.size,interpolation=self.method)
        
        
if __name__=='__main__':

    creator=create_tusimple_data(TUSIMPLE_PATH,5)
    creator()
    






    

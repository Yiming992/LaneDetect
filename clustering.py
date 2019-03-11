import numpy as np 
from sklearn.cluster import MeanShift,DBSCAN,estimate_bandwidth
import cv2

class lane_cluster():
    '''
    聚类算法，根据LaneNet Embedding 模块的输出，进行聚类已实现对单个车道线的检测
    '''
    def __init__(self,bandwidth,image,embedding,binary_mask,mode='line',method='Meanshift'):
        self.color=[np.array([255,0,0]),
                    np.array([0,255,0]),
                    np.array([0,0,255]),
                    np.array([125,125,0]),
                    np.array([0,125,125]),
                    np.array([125,0,125]),
                    np.array([50,100,50]),
                    np.array([100,50,100])]
        
        self.image=image
        self.bandwidth=bandwidth
        self.embedding=embedding
        self.binary=binary_mask
        self.mode=mode
        self.method=method

    def _get_lane_area(self):        
        idx=np.where(self.binary==1)
        lane_area=[]
        lane_idx=[]
        for i,j in zip(*idx):
            lane_area.append(self.embedding[:,i,j])
            lane_idx.append((i,j))
        return np.array(lane_area),lane_idx

    def _cluster(self,prediction):
        if self.method=='Meanshift':
            clustering=MeanShift(bandwidth=self.bandwidth,bin_seeding=True).fit(prediction)
        elif self.method=='DBSCAN':
            clustering=DBSCAN().fit(prediction)
        return clustering.labels_

    def _get_instance_masks(self):
        lane_area,lane_idx=self._get_lane_area()
        instance_mask=self.image

        labels=self._cluster(lane_area)
        num_cluster=len(set(labels))
        lane_idx=np.array(lane_idx)

        if not self.mode=='line':
            for index,label in enumerate(labels):
                instance_mask[lane_idx[index][0],lane_idx[index][1],:]=self.color[label]
            return instance_mask

        for index,label in enumerate(range(num_cluster)):
            pos=np.where(labels==label)
            coords=lane_idx[pos]
            coords=np.flip(coords,axis=1)
            coords=np.array([coords])

            color_map=(int(self.color[index][0]),
                       int(self.color[index][1]),
                       int(self.color[index][2]))
            cv2.polylines(instance_mask,coords,True,color_map,2)
        return instance_mask
   
    def __call__(self):
        return self._get_instance_masks()
        




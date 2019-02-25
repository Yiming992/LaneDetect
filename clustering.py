####Inference阶段的聚类算法
import numpy as np 
from sklearn.cluster import MeanShift,DBSCAN,estimate_bandwidth
import cv2
class lane_cluster():

    def __init__(self,bandwidth,embedding,binary_mask):
        self.color=[np.array([255,0,0]),
                    np.array([0,255,0]),
                    np.array([0,0,255]),
                    np.array([125,125,0]),
                    np.array([0,125,125]),
                    np.array([125,0,125]),
                    np.array([50,100,50]),
                    np.array([100,50,100])]
                    
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
        ms=MeanShift(bandwidth=self.bandwidth,bin_seeding=True)
        clustering=ms.fit(prediction)
        return clustering.cluster_centers_,clustering.labels_

    def _get_instance_masks(self):
        lane_area,lane_idx=self._get_lane_area()
        instance_mask=np.zeros((self.binary.shape[0],self.binary.shape[1],3))

            color_map=(int(self.color[index][0]),
                       int(self.color[index][1]),
                       int(self.color[index][2]))
            cv2.polylines(instance_mask,coords,True,color_map,2)
        return instance_mask
   
    def __call__(self):
        return self._get_instance_masks(self.embedding)
        




####Inference阶段的聚类算法
import numpy as np 
from sklearn.cluster import MeanShift,estimate_bandwidth
class lane_cluster():

    def __init__(self,bandwidth,image,embedding,binary_mask):
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

    def _get_lane_area(self):        
        idx=np.where(self.binary==1)
        lane_area=[]
        lane_idx=[]
        for i,j in zip(*idx):
            lane_area.append(self.embedding[:,i,j])
            lane_idx.append((i,j))
        return np.array(lane_area),lane_idx

    def _cluster(self,prediction):
        clustering=MeanShift(bandwidth=self.bandwidth,bin_seeding=True).fit(prediction)
        return clustering.cluster_centers_,clustering.labels_

    def _get_instance_masks(self):
        lane_area,lane_idx=self._get_lane_area()
        print(self.binary.shape)
        instance_mask=self.image

        centers,labels=self._cluster(lane_area)
        for index,label in enumerate(labels):
            instance_mask[lane_idx[index][0],lane_idx[index][1],:]=self.color[label]
        return instance_mask,labels
   
    def __call__(self):
        return self._get_instance_masks()
        




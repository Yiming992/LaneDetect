import numpy as np 
from sklearn.cluster import MeanShift,DBSCAN,estimate_bandwidth,mean_shift
import cv2
from sklearn.cluster import AffinityPropagation

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
        
        self.bandwidth=bandwidth
        self.image=image
        self.embedding=embedding
        self.binary=binary_mask
        self.mode=mode
        self.method=method

    def _get_lane_area(self):
        #print(self.binary)        
        idx=np.where(self.binary==1)
        lane_area=[]
        lane_idx=[]
        for i,j in zip(*idx):
            #print(self.embedding[:,i,j])
            lane_area.append(self.embedding[:,i,j])
            lane_idx.append((i,j))
        return np.array(lane_area),lane_idx

    def _cluster(self,prediction):
        if self.method=='Meanshift':
            clustering=MeanShift(bandwidth=self.bandwidth,bin_seeding=True,min_bin_freq=500,n_jobs=8).fit(prediction)
        elif self.method=='DBSCAN':
            clustering = DBSCAN(eps=.5,min_samples=1000).fit(prediction)
        return clustering.labels_#labelsclustering.labels

    def _get_instance_masks(self):
        lane_area,lane_idx=self._get_lane_area()
        lane_idx=np.array(lane_idx)
  
        image=self.image
        mask=np.zeros_like(image)
        #image=cv2.cvtColor(instance_mask,cv2.COLOR_RGB2BGRA)
        if len(lane_area.shape)!=2:
            return image
        labels=self._cluster(lane_area)
        print(lane_idx)
        
        _,unique_label=np.unique(labels,return_index=True)
        #unique_label=unique_label.tolist()
        unique_label=labels[np.sort(unique_label)]
        color_map={}
        for index,label in enumerate(unique_label):
            color_map[label]=index
        print(color_map)
        if not self.mode=='line':
            for index,label in enumerate(labels):
                mask[lane_idx[index][0],lane_idx[index][1],:]=self.color[color_map[label]]
            masked_image=cv2.addWeighted(image,1,mask,1,0)
            return masked_image
   
    def __call__(self):
        return self._get_instance_masks()
        




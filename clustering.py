import numpy as np 
from sklearn.cluster import MeanShift,DBSCAN
import cv2
from numpy.polynomial import polynomial as P
from collections import defaultdict

class lane_cluster():
    '''
    implement postprocess steps 
    '''
    def __init__(self,bandwidth,image,embedding,binary_mask,degree=3,method='Meanshift'):
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
        self.degree=degree
        self.method=method

    def _get_lane_area(self):      
        idx=np.where(self.binary.T==1)
        lane_area=[]
        lane_idx=[]
        for i,j in zip(*idx):
            lane_area.append(self.embedding[:,j,i])
            lane_idx.append((j,i))
        return np.array(lane_area),lane_idx

    def _cluster(self,prediction):
        if self.method=='Meanshift':
            clustering=MeanShift(bandwidth=self.bandwidth,bin_seeding=True,min_bin_freq=500,n_jobs=8).fit(prediction)
        elif self.method=='DBSCAN':
            clustering = DBSCAN(eps=.5,min_samples=1000).fit(prediction)
        return clustering.labels_

    def _get_instance_masks(self):
        lane_area,lane_idx=self._get_lane_area()
        lane_idx=np.array(lane_idx)
  
        image=self.image
        mask=np.zeros_like(image)
        segmentation_mask=np.zeros_like(image)
        if len(lane_area.shape)!=2:
            return image
        labels=self._cluster(lane_area)
        
        _,unique_label=np.unique(labels,return_index=True)
        unique_label=labels[np.sort(unique_label)]
        color_map={}
        polynomials=defaultdict(list)
        for index,label in enumerate(unique_label):
            color_map[label]=index
        for index,label in enumerate(labels):
            segmentation_mask[lane_idx[index][0],lane_idx[index][1],:]=self.color[color_map[label]]
            if len(polynomials[label])==0:
                polynomials[label].append([lane_idx[index][0],lane_idx[index][1]])
            elif 30>lane_idx[index][1]-polynomials[label][-1][1]>5:
                polynomials[label].append([lane_idx[index][0],lane_idx[index][1]])
        for label in polynomials.keys():
            c=P.polyfit(np.array(polynomials[label])[:,0],np.array(polynomials[label])[:,1],deg=3)
            max_x=max(np.array(polynomials[label])[:,0])
            min_x=min(np.array(polynomials[label])[:,0])
            for x_coor in range(min_x,max_x,5):
                y_coor=c[0]+c[1]*x_coor+c[2]*(x_coor**2)+c[3]*(x_coor**3)
                circle_color=self.color[color_map[label]]
                cv2.circle(mask,(int(y_coor),int(x_coor)),3,(int(circle_color[0]),int(circle_color[1]),int(circle_color[2])),-1)
        masked_image=cv2.addWeighted(image,0.2,mask,1,0)
        segmentation_mask=cv2.addWeighted(image,.9,segmentation_mask,1,0)
        return masked_image,segmentation_mask

    def __call__(self):
        return self._get_instance_masks()
        




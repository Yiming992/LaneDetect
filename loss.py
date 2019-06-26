import torch
import numpy as np 
import os
from collections import defaultdict
import torch.nn as nn
import cv2

class Losses:
    def __init__(self,batch,predictions,seg_mask,
                 embeddings,instance_mask,delta_v=.5,delta_d=3,
                 alpha=1,beta=1,gamma=0.001):
        '''
        Attributes:
            batch:batch
            predictions:LaneNet segmentation head outputs
            seg_mask:semantic segmentation ground-truth
            embeddings:Lanenet embedding head output
            instance_mask:instance segmentation ground truth
            delta_v:Variance loss hinge threshold
            delta_d:Distance loss hinge threshold
            alpha:Variance loss weight
            beta:Distance loss weight
            gamma:Regularization loss weight
        '''
        self.batch=batch
        self.predictions=predictions
        self.seg_mask=seg_mask
        self.embeddings=embeddings
        self.instance_mask=instance_mask
        self.delta_v=delta_v
        self.delta_d=delta_d
        self.alpha=alpha
        self.beta=beta 
        self.gamma=gamma

    def _bi_weight(self):
        frequency=defaultdict(lambda:torch.tensor(0.))
        for i in range(self.batch):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            img_tensor=self.seg_mask[i,:,:]
            frequency['background']+=(img_tensor==0.).sum()
            frequency['lane']+=(img_tensor==1.).sum()
        class_weights=defaultdict(lambda:torch.tensor(0.))
        class_weights['background']=1./torch.log(1.02+frequency['background']/(frequency['background']+frequency['lane']))
        class_weights['lane']=1./torch.log(1.02+frequency['lane']/(frequency['background']+frequency['lane']))
        return class_weights

    def _segmentation_loss(self):
        class_weights=self._bi_weight()
        loss=nn.CrossEntropyLoss(weight=torch.tensor([class_weights['background'].item(),class_weights['lane'].item()])).cuda()
        label=self.seg_mask.type(torch.long)
        loss=loss(self.predictions,label)
        return loss

    def _discriminative_loss(self):
        num_samples=self.instance_mask.size(0)
        dis_loss=torch.tensor(0.).cuda()
        var_loss=torch.tensor(0.).cuda()
        reg_loss=torch.tensor(0.).cuda()
        for i in range(num_samples):
            clusters=[]
            sample_embedding=self.embeddings[i,:,:,:]
            sample_label=self.instance_mask[i,:,:]
            num_clusters=len(sample_label.unique())-1
            vals=sample_label.unique()[1:]
            sample_label=sample_label.view(sample_label.size(0)*sample_label.size(1))
            sample_embedding=sample_embedding.view(-1,sample_embedding.size(1)*sample_embedding.size(2))
            v_loss=torch.tensor(0.).cuda()
            d_loss=torch.tensor(0.).cuda()
            r_loss=torch.tensor(0.).cuda()
            for j in range(num_clusters):
                indices=(sample_label==vals[j]).nonzero()
                indices=indices.squeeze()
                cluster_elements=torch.index_select(sample_embedding,1,indices)
                Nc=cluster_elements.size(1)
                mean_cluster=cluster_elements.mean(dim=1,keepdim=True)
                clusters.append(mean_cluster)
                v_loss+=torch.pow((torch.clamp(torch.norm(cluster_elements-mean_cluster)-self.delta_v,min=0.)),2).sum()/Nc
                r_loss+=torch.sum(torch.abs(mean_cluster))
            for index in range(num_clusters):
                for idx,cluster in enumerate(clusters):
                    if index==idx:
                        continue 
                    else:
                        distance=torch.norm(clusters[index]-cluster)#torch.sqrt(torch.sum(torch.pow(clusters[index]-cluster,2)))
                        d_loss+=torch.pow(torch.clamp(self.delta_d-distance,min=0.),2)
            var_loss+=v_loss/num_clusters
            dis_loss+=d_loss/(num_clusters*(num_clusters-1))
            reg_loss+=r_loss/num_clusters
        return self.alpha*(var_loss/num_samples)+self.beta*(dis_loss/num_samples)+self.gamma*(reg_loss/num_samples)

    def _total_loss(self):
        segmentation_loss=self._segmentation_loss()
        discriminative_loss=self._discriminative_loss()     
        total_loss=segmentation_loss+discriminative_loss
        return total_loss,segmentation_loss,discriminative_loss

    def __call__(self):
        return self._total_loss()







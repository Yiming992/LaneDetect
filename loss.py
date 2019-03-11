import torch
import numpy as np 
import os
from collections import defaultdict
import torch.nn as nn
import cv2

'''

'''
###
def bi_weight(data,batch):
    frequency=defaultdict(lambda:torch.tensor(0.))
    for i in range(batch):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        img_tensor=data[i,:,:]
        frequency['background']+=(img_tensor==0.).sum()
        frequency['lane']+=(img_tensor==1.).sum()
    class_weights=defaultdict(lambda:torch.tensor(0.))
    class_weights['background']=1./torch.log(1.02+frequency['background']/(frequency['background']+frequency['lane']))
    class_weights['lane']=1./torch.log(1.02+frequency['lane']/(frequency['background']+frequency['lane']))
    return class_weights

###语意分割损失函数
def Segmentation_loss(predictions,label,class_weights):
    loss=nn.CrossEntropyLoss(weight=torch.tensor([class_weights['background'].item(),class_weights['lane'].item()]).cuda())
    label=label.type(torch.long)
    loss=loss(predictions,label)
    return loss	

'''

'''
####聚类损失函数

def variance(delta_v,embeddings,labels):
    num_samples=labels.size(0)
    var_loss=torch.tensor(0.).cuda()
    for i in range(num_samples):
        sample_embedding=embeddings[i,:,:,:]
        sample_label=labels[i,:,:]
        num_clusters=len(sample_label.unique())-1
        vals=sample_label.unique()
        sample_label=sample_label.view(sample_label.size(0)*sample_label.size(1))
        sample_embedding=sample_embedding.view(-1,sample_embedding.size(1)*sample_embedding.size(2))
        loss=torch.tensor(0.).cuda()
        for j in range(num_clusters):
            indices=(sample_label==vals[j]).nonzero()
            indices=indices.squeeze()
            cluster_elements=torch.index_select(sample_embedding,1,indices)
            Nc=cluster_elements.size(1)
            mean_cluster=cluster_elements.mean(dim=1,keepdim=True)
            distance=torch.norm(cluster_elements-mean_cluster)
            loss+=torch.pow((torch.clamp(distance-delta_v,min=0.)),2).sum()/Nc
        var_loss+=loss/num_clusters
    return var_loss/num_samples

def distance(delta_d,embeddings,labels):
    num_samples=labels.size(0)
    dis_loss=torch.tensor(0.).cuda()
    for i in range(num_samples):
        clusters=[]
        sample_embedding=embeddings[i,:,:,:]
        sample_label=labels[i,:,:]
        num_clusters=len(sample_label.unique())-1
        vals=sample_label.unique()
        sample_label=sample_label.view(sample_label.size(0)*sample_label.size(1))
        sample_embedding=sample_embedding.view(-1,sample_embedding.size(1)*sample_embedding.size(2))
        loss=torch.tensor(0.).cuda()
        for j in range(num_clusters):
            indices=(sample_label==vals[j]).nonzero()
            indices=indices.squeeze()
            cluster_elements=torch.index_select(sample_embedding,1,indices)
            mean_cluster=cluster_elements.mean(dim=1)
            clusters.append(mean_cluster)
        for index in range(num_clusters):
            for idx,cluster in enumerate(clusters):
                if index==idx:
                    continue
                else:
                    distance=torch.norm(clusters[index]-cluster)
                    loss+=torch.pow(torch.clamp(delta_d-distance,min=0.),2)
        dis_loss+=loss/(num_clusters*(num_clusters-1))
    return dis_loss/num_samples

def reg(embeddings,labels):
    num_samples=labels.size(0)
    reg_loss=torch.tensor(0.).cuda()
    for i in range(num_samples):
        sample_embedding=embeddings[i,:,:,:]
        sample_label=labels[i,:,:]
        num_clusters=len(sample_label.unique())-1
        vals=sample_label.unique()
        sample_label=sample_label.view(sample_label.size(0)*sample_label.size(1))
        sample_embedding=sample_embedding.view(-1,sample_embedding.size(1)*sample_embedding.size(2))
        loss=torch.tensor(0.).cuda()
        for j in range(num_clusters):
            indices=(sample_label==vals[j]).nonzero()
            indices=indices.squeeze()
            cluster_elements=torch.index_select(sample_embedding,1,indices)
            mean_cluster=cluster_elements.mean(dim=1)
            euclidean=torch.sum(torch.abs(mean_cluster))
            if torch.isnan(euclidean):
                print(cluster_elements)
                print('labels:{},c:{}'.format(sample_label.unique(),num_clusters))
            loss+=euclidean
        reg_loss+=loss/num_clusters
    return reg_loss/num_samples

def instance_loss(delta_v,delta_d,embeddings,labels):
    variance_loss=variance(delta_v,embeddings,labels)
    distance_loss=distance(delta_d,embeddings,labels)
    reg_loss=reg(embeddings,labels)
    total_loss=variance_loss+distance_loss+.001*reg_loss
    return total_loss,variance_loss,distance_loss,reg_loss

class Losses:

    '''
    Implement above losses in object oriented way

    To be continued
    '''

    def __init__(self):
        pass

    def _bi_weight(self):
        pass

    def segmentation_loss(self):
        pass 

    def instance_loss(self):
        pass 

    def _variance(self):
        pass 
    
    def _distance(self):
        pass 
        
    def _reg(self):
        pass

    def _get_means(self):
        pass


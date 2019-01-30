import torch
import numpy as np 
import os
from collections import defaultdict
import torch.nn as nn
import cv2

###bounded inverse weights
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
    
####聚类损失函数
##[255,205,155,105,55,5]
def variance(delta_v,embeddings,labels):
    #vals=[255,205,155,105,55]
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
        #print('variance {}'.format(num_clusters))
        for j in range(num_clusters):
            indices=(sample_label==vals[j]).nonzero()
            indices=indices.squeeze()
            cluster_elements=torch.index_select(sample_embedding,1,indices)
            Nc=cluster_elements.size(1)
            mean_cluster=cluster_elements.mean(dim=1,keepdim=True)
            #print('v:{}'.format(mean_cluster))
            distance=torch.norm(cluster_elements-mean_cluster)#torch.sqrt(torch.sum(torch.pow(cluster_elements-mean_cluster,2),dim=0))
            loss+=torch.pow((torch.clamp(distance-delta_v,min=0.)),2).sum()/Nc
        var_loss+=loss/num_clusters
    return var_loss/num_samples

def distance(delta_d,embeddings,labels):
    #vals=[255,205,155,105,55]
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
        #print('distance {}'.format(num_clusters))
        for j in range(num_clusters):
            indices=(sample_label==vals[j]).nonzero()
            indices=indices.squeeze()
            cluster_elements=torch.index_select(sample_embedding,1,indices)
            mean_cluster=cluster_elements.mean(dim=1)
            #print('d:{}'.format(mean_cluster))
            clusters.append(mean_cluster)
        for index in range(num_clusters):
            for idx,cluster in enumerate(clusters):
                if index==idx:
                    continue
                else:
                    distance=torch.norm(clusters[index]-cluster)#torch.sqrt(torch.sum(torch.pow(clusters[index]-cluster,2)))
                    loss+=torch.pow(torch.clamp(delta_d-distance,min=0.),2)
        dis_loss+=loss/(num_clusters*(num_clusters-1))
    return dis_loss/num_samples

def reg(embeddings,labels):
    #vals=[255,205,155,105,55]
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
        #print('reg {}'.format(num_clusters))
        for j in range(num_clusters):
            indices=(sample_label==vals[j]).nonzero()
            indices=indices.squeeze()
            cluster_elements=torch.index_select(sample_embedding,1,indices)
            mean_cluster=cluster_elements.mean(dim=1)
            #print('mean:{}'.format(mean_cluster))
            euclidean=torch.sum(torch.abs(mean_cluster))#torch.sqrt(torch.dot(mean_cluster,mean_cluster))
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

###损失函数用于车道线拟合
def Hnet_loss():
    pass
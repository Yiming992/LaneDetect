import os 
import numpy as np 
import cv2
from model import LaneNet 
from torch.nn import DataParallel
from clustering import lane_cluster
import torch
import ffmpeg
import argparse 

class Test:
    def __init__(self,input_dir,output_dir,model_path,bandwidth,
                 mode='parallel',image_size=(256,512),threshold=.9):
        '''
        实现对未见输入数据的车道线划分
        变量：
           input_dir:输入数据所在文件夹
           output_dir:输出结果所在文件夹
           model_path:模型定义文件地址
           bandwidth:Meanshift 聚类所需参数
           mode:模型训练时的模式
           image_size:输入模型的图像尺寸
           thrshold:二向判断的阈值
        '''
        self.input_dir=input_dir
        self.output_dir=output_dir 
        self.model_path=model_path
        self.mode=mode
        self.image_size=image_size
        self.threshold=threshold
        self.bandwidth=bandwidth
    
    def _load_model(self):
        model=LaneNet()
        if self.mode=='parallel':
            model=DataParallel(model)
        model.load_state_dict(torch.load(self.model_path))
        model=model.cuda()
        return model
    
    def _frame_process(self,image_path,model):
        image=cv2.imread(os.path.join(self.input_dir,image_path),cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,self.image_size)
        img=image

        image=image.transpose(2,0,1)
        image=image[np.newaxis,:,:,:]
        image=image/255
        image=torch.tensor(image,dtype=torch.float)
        segmentation,embeddings=model(image.cuda())
        
        binary_mask=segmentation.data.cpu().numpy()
        binary_mask=binary_mask.squeeze()

        exp_binary_mask=np.exp(binary_mask)
        exp_binary_mask=exp_binary_mask.sum(axis=0)
        binary_mask=np.exp(binary_mask)/exp_binary_mask

        if not os.path.exists(os.path.join(self.output_dir,'binary/')):
            os.mkdir(os.path.join(self.output_dir,'binary/'))
        cv2.imwrite(os.path.join(self.output_dir,'binary/',image_path),binary_mask[1,:,:]*255)
        threshold_mask=binary_mask[1,:,:]>self.threshold
        threshold_mask=threshold_mask.astype(np.float)
        return embeddings,threshold_mask,img

    def img2img(self):
        model=self._load_model()
        model.eval()
        img_files=os.listdir(self.input_dir)
        for i in img_files:
            embeddings,threshold_mask,img=self._frame_process(i,model)
            cluster=lane_cluster(self.bandwidth,img,embeddings.squeeze().data.cpu().numpy(),
                                 threshold_mask,mode='point',method='Meanshift')
            instance_mask=cluster()
            if not os.path.exists(os.path.join(self.output_dir,'instance/')):
                os.mkdir(os.path.join(self.output_dir,'instance/'))
            instance_mask=cv2.cvtColor(instance_mask,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.output_dir,'instance/','.'.join([i.split('.')[0],'png'])),instance_mask)
            
    def img2video(self):
        pass

    def video2video(self):
        pass

    def video2img(self):
        pass 

if __name__=='__main__':

    args=argparse.ArgumentParser()

    args.add_argument('-i','--input',default='D:/Bjoy/LaneDetect/train_set/clips/0531/1492628585982117150')
    args.add_argument('-o','--output',default='./test_result')
    args.add_argument('-mp','--model',default='./logs/models/model_1_1552533346_100.pkl')
    args.add_argument('-m','--mode',default='parallel')
    args.add_argument('-s','--size',default=[256,512],type=int,nargs='+')
    args.add_argument('-t','--threshold',default=.9,type=float)
    args.add_argument('-b','--bandwidth',default=.7)
    
    args=args.parse_args()

    test=Test(args.input,args.output,args.model,args.bandwidth,mode=args.mode,image_size=tuple(args.size),threshold=args.threshold)
    test.img2img()







        



    
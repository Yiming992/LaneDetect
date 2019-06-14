import os 
import numpy as np 
import cv2
from model import LaneNet 
from torch.nn import DataParallel
from clustering import lane_cluster
import torch
import ffmpeg
import argparse
import matplotlib.pyplot as plt


class Test:
    def __init__(self,input_dir,output_dir,model_path,bandwidth,
                 mode='parallel',image_size=(512,256),threshold=.9):
        '''
        实现对未见输入数据的车道线划分
        变量：
           input_dir:输入数据所在文件夹
           output_dir:输出结果所在文件夹
           model_path:模型定义文件地址
           bandwidth:Meanshift 聚类所需参数
           mode:模型训练时的模式
           image_size:输入模型的图像尺寸
           threshold:二向判断的阈值
        '''
        self.input_dir=input_dir
        self.output_dir=output_dir 
        self.model_path=model_path
        self.bandwidth=bandwidth
        self.mode=mode
        self.image_size=image_size
        self.threshold=threshold
         
    def _load_model(self):
        model=LaneNet()
        if self.mode=='parallel':
            model=DataParallel(model)
        model.load_state_dict(torch.load(self.model_path))
        model=model.cuda()
        return model

    def _frame_process(self,image_path,model):
        #image=cv2.imread(os.path.join(self.input_dir,image_path),cv2.IMREAD_UNCHANGED)
        image=cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,self.image_size)
        img=image

        image=image.transpose(2,0,1)
        image=image[np.newaxis,:,:,:]
        image=image/255
        image=torch.tensor(image,dtype=torch.float)
        segmentation,embeddings=model(image.cuda())
                
        binary_mask=segmentation.data.cpu().numpy()
        binary_mask=binary_mask.squeeze() 
    
        exp_mask=np.exp(binary_mask-np.max(binary_mask,axis=0))
        binary_mask=exp_mask/exp_mask.sum(axis=0)

        # if not os.path.exists(os.path.join(self.output_dir,'binary/')):
        #     os.mkdir(os.path.join(self.output_dir,'binary/'))
        
        threshold_mask=binary_mask[1,:,:]>self.threshold
        threshold_mask=threshold_mask.astype(np.uint8)
        threshold_mask=threshold_mask*255
        kernel = np.ones((2,2),np.uint8)
        threshold_mask = cv2.erode(threshold_mask,kernel,iterations=2)
        kernel=np.ones((2,2),np.uint8)
        threshold_mask=cv2.dilate(threshold_mask,kernel,iterations=1)
        # cv2.imshow("img",threshold_mask)
        # cv2.waitKey(0)
        mask=cv2.connectedComponentsWithStats(threshold_mask, connectivity=8, ltype=cv2.CV_32S)
        output_mask=np.zeros(threshold_mask.shape,dtype=np.uint8)
        for label in np.unique(mask[1]):
            if label==0:
                continue
            labelMask = np.zeros(threshold_mask.shape, dtype="uint8")
            labelMask[mask[1] == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 200:
                output_mask = cv2.add(output_mask,labelMask)    
        #cv2.imwrite(os.path.join(self.output_dir,'binary/',image_path),output_mask)
        output_mask=output_mask.astype(np.float)/255
        return embeddings,output_mask,img

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

    def video2video(self):
        model=self._load_model()
        model.eval()
        video=cv2.VideoCapture(self.input_dir)
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        out=cv2.VideoWriter(self.output_dir,fourcc,25.0,(512,256))
        frame_count=0
        while True:
            ret,img=video.read()
            if not ret:
                break
            else:
                frame_count+=1
                embeddings,threshold_mask,img=self._frame_process(img,model)
                cluster=lane_cluster(self.bandwidth,img,embeddings.squeeze().data.cpu().numpy(),
                                     threshold_mask,mode='point',method='Meanshift')
                instance_mask=cluster()
                instance_mask=cv2.cvtColor(instance_mask,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join('./test_result/','.'.join([str(frame_count),'png'])),instance_mask)
                out.write(instance_mask)
        
if __name__=='__main__':

    args=argparse.ArgumentParser()

    args.add_argument('-i','--input',default="/home/yiming/Desktop/LaneDetect/train_set/clips/0601/1494452621490750551")            #'./train_set/clips/0313-1/1020')
    args.add_argument('-o','--output',default='./test_result')
    args.add_argument('-mp','--model',default='./logs/models/model_1_1560480517_50.pkl')
    args.add_argument('-m','--mode',default='parallel')
    args.add_argument('-s','--size',default=[512,256],type=int,nargs='+')
    args.add_argument('-t','--threshold',default=.7,type=float)
    args.add_argument('-b','--bandwidth',default=3)
    
    args=args.parse_args()

    test=Test(args.input,args.output,args.model,args.bandwidth,mode=args.mode,image_size=tuple(args.size),threshold=args.threshold)
    test.video2video()

    





        



    
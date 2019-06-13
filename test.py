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
    # def _morphological_process(self,image, kernel_size=5):
    #     """
    #     :param image:
    #     :param kernel_size:
    #     :return:
    #     """
    #     if image.dtype is not np.uint8:
    #         image = np.array(image, np.uint8)
    #     if len(image.shape) == 3:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    #     # close operation fille hole
    #     closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    #     return closing
    
    # def _connect_components_analysis(self,image):
    #     """
    #     :param image:
    #     :return:
    #     """
    #     if len(image.shape) == 3:
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray_image = image

    #     return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)
    
    # def _postprocess(self,image, minarea_threshold=15):
    #     """
    #     :param image:
    #     :param minarea_threshold: 连通域分析阈值
    #     :return:
    #     """
    #     # 首先进行图像形态学运算
    #     morphological_ret = self._morphological_process(image, kernel_size=5)

    #     # 进行连通域分析
    #     connect_components_analysis_ret = self._connect_components_analysis(image=morphological_ret)

    #     # 排序连通域并删除过小的连通域
    #     labels = connect_components_analysis_ret[1]
    #     stats = connect_components_analysis_ret[2]

    #     for index, stat in enumerate(stats):
    #         if stat[4] <= minarea_threshold:
    #             idx = np.where(labels == index)
    #             morphological_ret[idx] = 0

    #     return morphological_ret
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
    
        exp_mask=np.exp(binary_mask-np.max(binary_mask,axis=0))
        binary_mask=exp_mask/exp_mask.sum(axis=0)

        if not os.path.exists(os.path.join(self.output_dir,'binary/')):
            os.mkdir(os.path.join(self.output_dir,'binary/'))
        
        threshold_mask=binary_mask[1,:,:]>self.threshold
        threshold_mask=threshold_mask.astype(np.uint8)
        threshold_mask=threshold_mask*255
        kernel = np.ones((3,3),np.uint8)
        threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("img",threshold_mask)
        cv2.waitKey(0)
        mask=cv2.connectedComponentsWithStats(threshold_mask, connectivity=4, ltype=cv2.CV_32S)
        #output_mask=np.zerso(threshold_mask.shape,dtype=np.float)
        # for label in range(mask[0]):
        #     if label==0:
        #         continue
        output_mask=np.zeros(threshold_mask.shape,dtype=np.uint8)
        for label in np.unique(mask[1]):
            print(mask[1])
	# if this is the background label, ignore it
            if label==0:
                continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
            labelMask = np.zeros(threshold_mask.shape, dtype="uint8")
            labelMask[mask[1] == label] = 255
            numPixels = cv2.countNonZero(labelMask)
        
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
            if numPixels > 500:
                output_mask = cv2.add(output_mask,labelMask)
        #_,mask=cv2.threshold(mask[1].astype(np.uint8),1,255,cv2.THRESH_BINARY)     
        cv2.imwrite(os.path.join(self.output_dir,'binary/',image_path),output_mask)
        
        #threshold_mask=cv2.connectedComponentsWithStats(threshold_mask, connectivity=8, ltype=cv2.CV_32S)
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

    def img2video(self):
        pass 

    def video2video(self):
        pass 

    def video2img(self):
        pass 

if __name__=='__main__':

    args=argparse.ArgumentParser()

    args.add_argument('-i','--input',default="/home/yiming/Desktop/LaneDetect/train_set/clips/0601/1494452621490750551")            #'./train_set/clips/0313-1/1020')
    args.add_argument('-o','--output',default='./test_result')
    args.add_argument('-mp','--model',default='./logs/models/model_1_1560249080_174.pkl')
    args.add_argument('-m','--mode',default='parallel')
    args.add_argument('-s','--size',default=[512,256],type=int,nargs='+')
    args.add_argument('-t','--threshold',default=.9,type=float)
    args.add_argument('-b','--bandwidth',default=2.5)
    
    args=args.parse_args()

    test=Test(args.input,args.output,args.model,args.bandwidth,mode=args.mode,image_size=tuple(args.size),threshold=args.threshold)
    test.img2img()

    





        



    
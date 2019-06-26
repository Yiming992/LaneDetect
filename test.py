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
    def __init__(self,input_address,output_address,model_path,bandwidth,
                 mode='parallel',image_size=(512,256),threshold=.9):

        self.input_ad=input_address
        self.output_ad=output_address 
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

    def _load_images(self,image_file):
        image_frame=cv2.imread(os.path.join(self.input_ad,image_file),cv2.IMREAD_UNCHANGED)
        return image_frame

    def _frame_process(self,image_frame,model):
        image=cv2.cvtColor(image_frame,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,self.image_size,interpolation=cv2.INTER_NEAREST)
        img=image.copy()

        image=image.transpose(2,0,1)
        image=image[np.newaxis,:,:,:]
        image=image/255
        image=torch.tensor(image,dtype=torch.float)
        segmentation,embeddings=model(image.cuda())
                
        binary_mask=segmentation.data.cpu().numpy()
        binary_mask=binary_mask.squeeze() 
    
        exp_mask=np.exp(binary_mask-np.max(binary_mask,axis=0))
        binary_mask=exp_mask/exp_mask.sum(axis=0)
        threshold_mask=binary_mask[1,:,:]>self.threshold
        threshold_mask=threshold_mask.astype(np.uint8)
        threshold_mask=threshold_mask*255
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(4, 4))
        threshold_mask = cv2.dilate(threshold_mask,kernel,iterations=1)
        mask=cv2.connectedComponentsWithStats(threshold_mask, connectivity=8, ltype=cv2.CV_32S)
        output_mask=np.zeros(threshold_mask.shape,dtype=np.uint8)
        for label in np.unique(mask[1]):
            if label==0:
                continue
            labelMask = np.zeros(threshold_mask.shape, dtype="uint8")
            labelMask[mask[1] == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 400:
                output_mask = cv2.add(output_mask,labelMask)    
        output_mask=output_mask.astype(np.float)/255
        return embeddings,output_mask,img

    def test_img(self):
        model=self._load_model()
        model.eval()
        img_files=os.listdir(self.input_ad)
        for i in img_files:
            img_frame=self._load_images(i)
            embeddings,threshold_mask,img=self._frame_process(img_frame,model)
            cluster=lane_cluster(self.bandwidth,img,embeddings.squeeze().data.cpu().numpy(),
                                 threshold_mask,method='Meanshift')
            fitted_image,instance_mask,=cluster()
            if not os.path.exists(os.path.join(self.output_ad,'instance/')):
                os.mkdir(os.path.join(self.output_ad,'instance/'))
            if not os.path.exists(os.path.join(self.output_ad,"fitted/")):
                os.mkdir(os.path.join(self.output_ad,"fitted/"))
            instance_mask=cv2.cvtColor(instance_mask,cv2.COLOR_RGB2BGR)
            fitted_image=cv2.cvtColor(fitted_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.output_ad,'instance/','.'.join([i.split('.')[0],'png'])),instance_mask)
            cv2.imwrite(os.path.join(self.output_ad,'fitted/','.'.join([i.split('.')[0],'png'])),fitted_image)

    def test_video(self):
        model=self._load_model()
        model.eval()
        video=cv2.VideoCapture(self.input_ad)
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        out=cv2.VideoWriter(self.output_ad,fourcc,25.0,(512,256))
        frame_count=0
        while True:
            ret,img=video.read()
            if not ret:
                break
            else:
                if cv2.waitKey(25)& 0xFF==ord("q"):
                    cv2.destroyAllWindows()
                    break
                frame_count+=1
                embeddings,threshold_mask,img=self._frame_process(img,model)
                cluster=lane_cluster(self.bandwidth,img,embeddings.squeeze().data.cpu().numpy(),
                                     threshold_mask,method='Meanshift')
                fitted_image,instance_mask=cluster()
                if not os.path.exists(os.path.join(self.output_ad,'instance/')):
                    os.mkdir(os.path.join(self.output_ad,'instance/'))
                if not os.path.exists(os.path.join(self.output_ad,"fitted/")):
                    os.mkdir(os.path.join(self.output_ad,"fitted/"))
                instance_mask=cv2.cvtColor(instance_mask,cv2.COLOR_RGB2BGR)
                fitted_image=cv2.cvtColor(fitted_image,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.output_ad,'instance/','.'.join([i.split('.')[0],'png'])),instance_mask)
                cv2.imwrite(os.path.join(self.output_ad,'fitted/','.'.join([i.split('.')[0],'png'])),fitted_image)
                out.write(instance_mask)
        
if __name__=='__main__':

    args=argparse.ArgumentParser()

    args.add_argument('-i','--input',default="./test_images")
    args.add_argument('-o','--output',default='./test_result')
    args.add_argument('-mp','--model',default='./logs/models/model_1_1560853544_210.pkl')
    args.add_argument('-m','--mode',default='parallel')
    args.add_argument('-s','--size',default=[512,256],type=int,nargs='+')
    args.add_argument('-t','--threshold',default=.5,type=float)
    args.add_argument('-b','--bandwidth',default=3)
    
    args=args.parse_args()

    test=Test(args.input,args.output,args.model,args.bandwidth,mode=args.mode,image_size=tuple(args.size),threshold=args.threshold)
    test.test_img()

    





        



    
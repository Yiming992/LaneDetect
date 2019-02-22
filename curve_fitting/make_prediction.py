from HNet import HNet 
from hnet_data import Hnet_Data 
import argparse
import torch
import cv2

def create_prediction(model,image):
    image=image.cuda()
    coefficient=model(image)
    pass

if __name__=='__main__':

    ap=argparse.ArgumentParser()

    ap.add_argument('-m','--model_path')
    ap.add_argument('-i','--image_path')
    ap.add_argument('-s','--save_path')
    
    args=vars(ap.parse_args())
    model=HNet()
    model.load_state_dict(torch.load(args['model_path']))
    model.eval()

    data_read=Hnet_Data(args['image_path'],None)
    image=data_read._imread(args['image_path'])



    



    

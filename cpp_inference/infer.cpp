#include "infer.hpp"

using namespace inference;

Mat Input_Reader::read(string f){
    Mat img_array=imread(f,IMREAD_UNCHANGED);
    Mat img_rgb;
    cvtColor(img_array,img_rgb,COLOR_BGR2RGB);
    return img_rgb;    
}

Mat Input_Reader::process(int w,int h,Mat t){
    Mat resized_img (w,h,3);
    resize(t,resized_img,resized_img.size());
    return resized_img;
}

void NV_rt::onnx2rt(){

}

void NV_rt::doinference(){

}



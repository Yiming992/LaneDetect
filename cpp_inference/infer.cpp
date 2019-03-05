#include "infer.hpp"
#include "common.h"
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"

using namespace inference;
using namespace std; 
using namespace cv;
using namespace nvinfer1;

static Logger gLogger;

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

void NV_rt::onnx2rt(const string& modelFile,
                    unsigned int batch_size,
                    IHostMemory*& trtModelStream){

                        int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
                        IBuilder* builder=createInferBuilder(gLogger);
                        nvinfer1::INetworkDefinition* network=builder->createNetwork();

                        auto parser = nvonnxparser::createParser(*network,gLogger);
}

void NV_rt::doinference(){

}



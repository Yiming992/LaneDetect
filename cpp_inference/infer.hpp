#ifndef INFER_H
#define INFER_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "Nvinfer.h"

namespace inference {

    class Input_Reader {
        int width;
        int height;
        std::string mode;
        std::string filepath;
       
        public:
            cv::Mat read(std::string);
            cv::Mat process(int,int,cv::Mat);
            Input_Reader(int w, int h, std::string m,std::string f){
                width=w;
                height=h;
                filepath=f;
                mode=m;
            }
    };
    class NV_rt{

        public:
            void onnx2rt(const string& modelFile,
                         unsigned int  batch_size,
                         nvinfer1::IHostMemory*& trtModelStream);

            void doinference();

    };

    class Clustering{

    };
}



#endif

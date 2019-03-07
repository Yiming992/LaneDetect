#ifndef INFER_H
#define INFER_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Nvinfer.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/mean_shift/mean_shift.hpp"
#include "common.h"
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include <assert.h>
#include <time.h>
#include <sys/stat.h>

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
            std::string locatefile(std::string);

            void onnx2rt(std::string,unsigned int,nvinfer1::IHostMemory*);

            void doinference();

    };

    class Clustering{
        public:
            struct cluster ();

    };
}
#endif

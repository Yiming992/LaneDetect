#ifndef INFER_H
#define INFER_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "Nvinfer.h"


using namespace std; 
using namespace cv;
using namespace nvinfer1;

namespace inference {

    class Input_Reader {
        int width;
        int height;

        string filepath;
        
        Mat read(string);

        Mat process(int,int,Mat);
        
        public:
            Input_Reader(int w, int h, string f){
                width=w;
                height=h;
                filepath=f;
            }
    };
    class NV_rt{

        public:
            void onnx2rt();

            void doinference();

    };

    class Clustering{

    };
}



#endif

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(){

    Mat I=imread("./data/LaneImages/");
    Mat target;
    resize(I,target,Size(256,512));
    imshow("target",target); 
}


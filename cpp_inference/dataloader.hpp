#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <iostream>
#include <string>
#include <torch/torch.h>
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;
using namespace torch

class Dataloader
{
  public:
    string file_path; 
    int width;
    int height;

    Mat preprocess(string,string);
    Tensor input_convert(Mat);
}
#endif 
#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std ;
using namespace cv;

class Dataloader
{
  public:
    string file_path; 
    int width;
    int height;

    Mat preprocess(string s);
}
#endif 
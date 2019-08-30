#pragma once
# ifndef TARGETFEATURE_H_
# define TARGETFEATURE_H_

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

class ColorFeature
{
private:
	Mat _img;
	int _number;
	vector<float> color_vector;
	Mat splited_img;
	int result;
public:
	ColorFeature();
	ColorFeature(Mat _img, int _number);
	Mat split_interval(Mat &img);
	vector<float> color_extract();
	~ColorFeature();
};

class HogFeature
{
private:
	Mat _img;
	Size _windowSize;
	Size blockSize;
	Size cellSize;
	Size blockStrideSize;
	int nbins;
public:
	HogFeature();
	HogFeature(Mat _img, Size _windowSize);
	~HogFeature();
	vector<float> hog_extract();
};

# endif
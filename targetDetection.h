#pragma once
# ifndef TARGETDETECTION_H_
# define TARGETDETECTION_H_

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

class PreProcess
{
private:
	Mat _src_img, HSV_img, bin_img;
	Size Gaussian_kernel;
	Mat Dilation_kernel, Erosion_kernel;
	int vmin, smax, vmax;
public:
	PreProcess();
	PreProcess(Mat _img);
	Mat pre_process(const string &name);
	Mat filter(Mat &img);
	~PreProcess();
};


class ContoursDet :public PreProcess
{
private:
	Mat _src_img;
	Mat bin_img;
	double k;
	double perimeter;
	vector<Rect> *result_contours = new vector<Rect>;
public:
	ContoursDet();
	ContoursDet(Mat _img);
	vector<Rect> contours_detecte(const string &name, bool is_show = false, int min_perimeter = 200, int max_perimeter = 600, double min_k = 0.5, double max_k = 1.5);
	void show_contour_result(Mat &img, vector<Rect> *result_contours);
	~ContoursDet();
};
# endif
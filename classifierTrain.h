#pragma once
# ifndef CLASSIFIERTRAIN_H_
# define CLASSIFIERTRAIN_H_

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

class ClassifierTrain 
{
private:
	Size windowSize;
	int color_numbers = 16;

	vector<int> *labels = new vector<int>; 
	vector<Rect> *result_contours = new vector<Rect>;
	vector<Rect> *new_rect = new vector<Rect>; 
	Mat src_img;

	ofstream out_file;
	vector<String> img_files;     // 存放文件路径  
	vector<String> label_files;   // 存放文件路径  
	size_t img_numbers;

	vector<float> color_vector;
	vector<float> hog_vector;
	vector<float> result_vector;

public:
	ClassifierTrain();
	double round(double number, int bits);
	vector<int> parse_xml(const string &strXmlPath);
	vector<Rect> reshape_ballRect(vector<int> *labels);
	vector<int> contours_to_labels(Rect contours);
	vector<float> get_result_vector(Mat &img, vector<Rect> *new_rect);

	void cal_pos_vector(const string &img_files_dir, const string &label_files_dir, const string &save_file_name);
	void cal_neg_vector(const string &img_files_dir, const string &save_file_name);
	void result_test(const string &img_files_dir);
	~ClassifierTrain();
};
# endif
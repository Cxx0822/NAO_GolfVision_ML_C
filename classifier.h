#pragma once
# ifndef CLASSIFIER_H_
# define CLASSIFIER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

struct DataSet {
	Mat data_train;
	Mat labels_train;
};

class Classifier
{
private:
	string _filename;
	DataSet dataSet;
	Ptr<LogisticRegression> lr = LogisticRegression::create();
	vector<int> result;
public:
	Classifier();
	Classifier(const string &filename);
	~Classifier();
	DataSet load_dataset();
	Ptr<LogisticRegression> classify_train(DataSet &dataSet);
	int  classify_result(Mat &result_mat);
};

# endif
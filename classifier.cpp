#include "classifier.h"

Classifier::Classifier()
{

}

Classifier::Classifier(const string &filename)
{
	this->_filename = filename;
}

Classifier::~Classifier()
{
}

DataSet Classifier::load_dataset()
{
	Mat data, labels;
	{
		FileStorage f;
		if (f.open(this->_filename, FileStorage::READ))
		{
			f["datamat"] >> data;
			f["labelsmat"] >> labels;
			f.release();
		}
		else
		{
			cerr << "file can not be opened: " << this->_filename << endl;
		}
		data.convertTo(data, CV_32F);
		labels.convertTo(labels, CV_32F);
	}

	for (int i = 0; i < data.rows; i++)
	{
		dataSet.data_train.push_back(data.row(i));
		dataSet.labels_train.push_back(labels.row(i));
	}

	return dataSet;
}


Ptr<LogisticRegression> Classifier::classify_train(DataSet &dataSet)
{
	cout << "training..." << endl;
	//! [init]
	lr->setLearningRate(0.001);
	lr->setIterations(10);
	lr->setRegularization(LogisticRegression::REG_L2);
	lr->setTrainMethod(LogisticRegression::BATCH);
	lr->setMiniBatchSize(1);

	lr->train(dataSet.data_train, ROW_SAMPLE, dataSet.labels_train);
	cout << "done!" << endl;

	return lr;
}

int  Classifier::classify_result(Mat &result_mat)
{
	Mat responses;
	lr->predict(result_mat, responses);

	result = (vector<int>)(responses.reshape(1, 1));

	return result.at(0);
}
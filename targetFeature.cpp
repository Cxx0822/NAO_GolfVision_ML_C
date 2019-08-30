#include "targetFeature.h"

/*��ɫ������ȡ�ඨ��*/
ColorFeature::ColorFeature()
{
}

ColorFeature::ColorFeature(Mat img, int number)
{
	this->_img = img;
	this->_number = number;
}


ColorFeature::~ColorFeature()
{
}

/*��������*/
Mat ColorFeature::split_interval(Mat &img)
{
	Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = img.end<Vec3b>();

	// ��ÿ�����ػ��ֵ����Ե�����
	for (; it != itend; ++it)
	{
		(*it)[0] = (*it)[0] / this->_number;     // ������ֵ20��20/16=1����������Ϊ1
		(*it)[1] = (*it)[1] / this->_number;
		(*it)[2] = (*it)[2] / this->_number;
	}

	return img;
}

/*��ɫ������ȡ*/
vector<float> ColorFeature::color_extract()
{
	vector<Mat> channels;
	vector<int> temp_channel;

	splited_img = split_interval(_img);
	split(splited_img, channels);        // ����ͨ��

										 // ����3��ͨ�����ֱ�ͳ�Ƹ��������������
	for (int i = 0; i < channels.size(); i++)
	{
		temp_channel = channels.at(i).reshape(1, 1);   // תΪvector
		for (int j = 0; j < this->_number; j++)
		{
			result = int(count(temp_channel.begin(), temp_channel.end(), j));   // ͳ�Ƴ��ֵĴ���
			color_vector.push_back(float(result) / temp_channel.size());       // ������ʣ�0-1��
		}
	}

	// ������ά��Ϊ48����16*3��ÿ��ͨ��16�����䡣
	return color_vector;
}


/*HOG������ȡ�ඨ��*/
HogFeature::HogFeature()
{
}

HogFeature::HogFeature(Mat img, Size windowSize)
{
	_img = img;
	_windowSize = windowSize;
}

HogFeature::~HogFeature()
{
}

/*HOG������ȡ*/
vector<float> HogFeature::hog_extract()
{
	Mat img_gray;

	// ���ڴ�С=���С=��������ֻ���һ�Σ���ԭͼ����������ͳһ����ά��
	// ��������Ϊ 4 * 8 = 32�� 4����Ϊ���С��Ԫ����С��4����8����Ϊbins=8
	blockSize = this->_windowSize;            // ���С
	cellSize = blockSize / 2;          // Ԫ����С
	blockStrideSize = blockSize;       // ����
	nbins = 8;                         // ֱ��ͼbin������

	HOGDescriptor hog(this->_windowSize, blockSize, blockStrideSize, cellSize, nbins);   // ʵ����hog��
	vector<float> hog_vector;                  // HOG����������
	cvtColor(this->_img, img_gray, CV_BGR2GRAY);      // HOG��Ҫ�Ҷ�ͼ
	hog.compute(img_gray, hog_vector, Size(4, 4), Size(0, 0));         // ����hog��������

	return hog_vector;
}
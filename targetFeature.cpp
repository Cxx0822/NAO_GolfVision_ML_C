#include "targetFeature.h"

/*颜色特征提取类定义*/
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

/*分离区间*/
Mat ColorFeature::split_interval(Mat &img)
{
	Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = img.end<Vec3b>();

	// 将每个像素划分到各自的区间
	for (; it != itend; ++it)
	{
		(*it)[0] = (*it)[0] / this->_number;     // 例像素值20，20/16=1，即区间数为1
		(*it)[1] = (*it)[1] / this->_number;
		(*it)[2] = (*it)[2] / this->_number;
	}

	return img;
}

/*颜色特征提取*/
vector<float> ColorFeature::color_extract()
{
	vector<Mat> channels;
	vector<int> temp_channel;

	splited_img = split_interval(_img);
	split(splited_img, channels);        // 分离通道

										 // 遍历3个通道，分别统计个数，并计算概率
	for (int i = 0; i < channels.size(); i++)
	{
		temp_channel = channels.at(i).reshape(1, 1);   // 转为vector
		for (int j = 0; j < this->_number; j++)
		{
			result = int(count(temp_channel.begin(), temp_channel.end(), j));   // 统计出现的次数
			color_vector.push_back(float(result) / temp_channel.size());       // 计算概率（0-1）
		}
	}

	// 总特征维度为48，即16*3，每个通道16个区间。
	return color_vector;
}


/*HOG特征提取类定义*/
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

/*HOG特征提取*/
vector<float> HogFeature::hog_extract()
{
	Mat img_gray;

	// 窗口大小=块大小=步长，即只检测一次，即原图，这样方便统一特征维度
	// 总特征数为 4 * 8 = 32， 4是因为块大小是元胞大小的4倍，8是因为bins=8
	blockSize = this->_windowSize;            // 块大小
	cellSize = blockSize / 2;          // 元胞大小
	blockStrideSize = blockSize;       // 步长
	nbins = 8;                         // 直方图bin的数量

	HOGDescriptor hog(this->_windowSize, blockSize, blockStrideSize, cellSize, nbins);   // 实例化hog类
	vector<float> hog_vector;                  // HOG描述子向量
	cvtColor(this->_img, img_gray, CV_BGR2GRAY);      // HOG需要灰度图
	hog.compute(img_gray, hog_vector, Size(4, 4), Size(0, 0));         // 计算hog特征向量

	return hog_vector;
}
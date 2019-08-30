#include "targetDetection.h"

PreProcess::PreProcess()
{
}

PreProcess::PreProcess(Mat img)
{
	this->_src_img = img;
}

PreProcess::~PreProcess()
{
}

/*转到HSV空间，并二值化处理*/
Mat PreProcess::pre_process(const string &name)
{
	if (this->_src_img.empty())
	{
		printf("图片加载失败\n");
	}
	else
	{
		cvtColor(this->_src_img, HSV_img, COLOR_BGR2HSV); // 转到HSV
		if (name == "football")
		{
			vmin = 41, smax = 34, vmax = 255;
			inRange(HSV_img, Scalar(0, 0, vmin), Scalar(180, smax, vmax), bin_img);   // 二值化
			bin_img = filter(bin_img);                                                // 滤波
		}
	}

	return bin_img;
}

/*图像形态学处理*/
Mat PreProcess::filter(Mat &img)
{
	Mat new_img;

	// 形态学核大小
	Gaussian_kernel = Size(9, 9);
	Dilation_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Erosion_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	erode(img, new_img, Dilation_kernel);                      // 腐蚀
	dilate(new_img, new_img, Dilation_kernel);                 // 膨胀  
	GaussianBlur(new_img, new_img, Gaussian_kernel, 0, 0);     // 高斯滤波

	return new_img;
}


/*轮廓检测类定义*/

ContoursDet::ContoursDet()
{
}

ContoursDet::ContoursDet(Mat img) : PreProcess(img)
{
	this->_src_img = img;   //将原图保存
}


ContoursDet::~ContoursDet()
{
	delete result_contours;
}

/*轮廓检测*/
vector<Rect> ContoursDet::contours_detecte(const string &name, bool is_show, int min_perimeter, int max_perimeter, double min_k, double max_k)
{
	bin_img = pre_process(name);

	vector<vector<Point>> contours;             //轮廓
	vector<Vec4i> hierarchy;                    //层次结构


	findContours(bin_img, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);   // 轮廓检测

	vector<Rect> boundRect(contours.size());

	//遍历每个轮廓
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		int x = boundRect[i].x, y = boundRect[i].y, w = boundRect[i].width, h = boundRect[i].height;
		k = double(h) / w;
		perimeter = arcLength(contours[i], true);

		//判断是否符合条件
		if ((perimeter > min_perimeter && perimeter < max_perimeter) && (k > min_k && k < max_k))
		{
			result_contours->push_back(boundRect[i]);
		}
	}

	if (is_show == true)
	{
		show_contour_result(bin_img, result_contours);
	}

	return *result_contours;
}

/*显示轮廓结果*/
void ContoursDet::show_contour_result(Mat &img, vector<Rect> *result_contours)
{
	for (int index = 0; index < result_contours->size(); index++)
	{
		rectangle(img, result_contours->at(index), Scalar(255, 255, 255), 2);
	}

	imshow("dst_image", img);
	waitKey(0);
	destroyAllWindows();
}


/*   C++不能在类中使用回调函数
void PreProcess::sliderObjectHSV()
{
namedWindow("football", WINDOW_AUTOSIZE);
createTrackbar("vmin", "football", &g_nVmin, 255, on_TrackbarHSV);
createTrackbar("smax:", "football", &g_nSmax, 255, on_TrackbarHSV);

on_TrackbarHSV(g_nVmin, 0);
on_TrackbarHSV(g_nSmax, 0);
waitKey(0);
}

void PreProcess::on_TrackbarHSV(int, void *)
{
inRange(HSV_img, Scalar(0, 0, g_nVmin), Scalar(180, g_nSmax, vmax), bin_img);
imshow("bin", bin_img);
}
*/
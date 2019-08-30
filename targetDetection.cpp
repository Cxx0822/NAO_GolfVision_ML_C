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

/*ת��HSV�ռ䣬����ֵ������*/
Mat PreProcess::pre_process(const string &name)
{
	if (this->_src_img.empty())
	{
		printf("ͼƬ����ʧ��\n");
	}
	else
	{
		cvtColor(this->_src_img, HSV_img, COLOR_BGR2HSV); // ת��HSV
		if (name == "football")
		{
			vmin = 41, smax = 34, vmax = 255;
			inRange(HSV_img, Scalar(0, 0, vmin), Scalar(180, smax, vmax), bin_img);   // ��ֵ��
			bin_img = filter(bin_img);                                                // �˲�
		}
	}

	return bin_img;
}

/*ͼ����̬ѧ����*/
Mat PreProcess::filter(Mat &img)
{
	Mat new_img;

	// ��̬ѧ�˴�С
	Gaussian_kernel = Size(9, 9);
	Dilation_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Erosion_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	erode(img, new_img, Dilation_kernel);                      // ��ʴ
	dilate(new_img, new_img, Dilation_kernel);                 // ����  
	GaussianBlur(new_img, new_img, Gaussian_kernel, 0, 0);     // ��˹�˲�

	return new_img;
}


/*��������ඨ��*/

ContoursDet::ContoursDet()
{
}

ContoursDet::ContoursDet(Mat img) : PreProcess(img)
{
	this->_src_img = img;   //��ԭͼ����
}


ContoursDet::~ContoursDet()
{
	delete result_contours;
}

/*�������*/
vector<Rect> ContoursDet::contours_detecte(const string &name, bool is_show, int min_perimeter, int max_perimeter, double min_k, double max_k)
{
	bin_img = pre_process(name);

	vector<vector<Point>> contours;             //����
	vector<Vec4i> hierarchy;                    //��νṹ


	findContours(bin_img, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);   // �������

	vector<Rect> boundRect(contours.size());

	//����ÿ������
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		int x = boundRect[i].x, y = boundRect[i].y, w = boundRect[i].width, h = boundRect[i].height;
		k = double(h) / w;
		perimeter = arcLength(contours[i], true);

		//�ж��Ƿ��������
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

/*��ʾ�������*/
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


/*   C++����������ʹ�ûص�����
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
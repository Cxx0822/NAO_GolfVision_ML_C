#include "classifierTrain.h"
#include "classifier.h"
#include "targetDetection.h"
#include "targetFeature.h"
#include "tinystr.h"  
#include "tinyxml.h"

ClassifierTrain::ClassifierTrain()
{
}

ClassifierTrain::~ClassifierTrain()
{
	delete labels;
	delete result_contours;
	delete new_rect;
}

/*将浮点数保留几位有效小数*/
double ClassifierTrain::round(double number, int bits)
{
	stringstream ss;
	ss << fixed << setprecision(bits) << number;
	ss >> number;

	return number;
}

/*解析xml文件，从https://www.sourceforge.net/projects/tinyxml/下载源代码，并将
tinystr.h、tinystr.cpp、tinyxml.h、tinyxml.cpp、tinyxmlerror.cpp、tinyxmlparser.cpp放入工程文件夹中
*/
vector<int> ClassifierTrain::parse_xml(const string &strXmlPath)
{


	//读取xml文件中的参数值
	TiXmlDocument* Document = new TiXmlDocument();
	if (!Document->LoadFile(strXmlPath.c_str()))
	{
		cout << "无法加载xml文件！" << endl;
	}
	TiXmlElement* RootElement = Document->RootElement();		    //根目录

	TiXmlElement* NextElement = RootElement->FirstChildElement();  //根目录下的第一个节点层

	while (NextElement != NULL)		//判断有没有读完
	{
		if (NextElement->ValueTStr() == "object")		//读到object节点
		{
			//NextElement = NextElement->NextSiblingElement();
			TiXmlElement* BoxElement = NextElement->FirstChildElement();
			while (BoxElement->ValueTStr() != "bndbox")		//读到box节点
			{
				BoxElement = BoxElement->NextSiblingElement();
			}
			//索引到xmin节点
			TiXmlElement* xminElemeng = BoxElement->FirstChildElement();
			{
				//分别读取四个数值
				labels->push_back(int(atof(xminElemeng->GetText())));
				TiXmlElement* yminElemeng = xminElemeng->NextSiblingElement();
				labels->push_back(int(atof(yminElemeng->GetText())));
				TiXmlElement* xmaxElemeng = yminElemeng->NextSiblingElement();
				labels->push_back(int(atof(xmaxElemeng->GetText())));
				TiXmlElement* ymaxElemeng = xmaxElemeng->NextSiblingElement();
				labels->push_back(int(atof(ymaxElemeng->GetText())));
			}
		}
		NextElement = NextElement->NextSiblingElement();
	}

	//释放内存
	delete Document;
	return *labels;
}

/*将labels信息转为rect信息*/
vector<Rect> ClassifierTrain::reshape_ballRect(vector<int> *labels)
{
	int x1 = labels->at(0), y1 = labels->at(1), x2 = labels->at(2), y2= labels->at(3);
	int w = int(x2 - x1), h = int(y2 - y1);

	// 保证w,h是偶数，如果不是，减小1个像素，也不影响结果
	if (int(w / 2) % 2 == 0)   w = int(w / 2);
	else                       w = int(w / 2) - 1;
	
	if (int(h / 2) % 2 == 0)   h = int(h / 2);
	else                       h = int(h / 2) - 1;

	new_rect->push_back(Rect(int(x1),     int(y1),     w, h));
	new_rect->push_back(Rect(int(x1 + w), int(y1),	   w, h));
	new_rect->push_back(Rect(int(x1),     int(y1 + h), w, h));
	new_rect->push_back(Rect(int(x1 + w), int(y1 + h), w, h));

	return *new_rect;
}

/*轮廓信息转为labels信息*/
vector<int> ClassifierTrain::contours_to_labels(Rect contours)
{
	vector<int> result_labels;

	result_labels.push_back(contours.x);
	result_labels.push_back(contours.y);
	result_labels.push_back(contours.x + contours.width);
	result_labels.push_back(contours.y + contours.height);

	return result_labels;
}

vector<float> ClassifierTrain::get_result_vector(Mat &img, vector<Rect> *new_rect)
{
	// 遍历每个小矩形
	for (int i = 0; i < new_rect->size(); i++)
	{
		Mat new_rect_area = img(new_rect->at(i));       // 得到小矩形的区域

		// 获取颜色特征
		ColorFeature colFea(new_rect_area, color_numbers);
		color_vector = colFea.color_extract();
		//cout << "color_extract.size: " << color_vector.size() << endl;

		// 获取HOG特征
		windowSize = Size(new_rect->at(i).width - 2, new_rect->at(i).height - 2);
		HogFeature hogFea(new_rect_area, windowSize);
		hog_vector = hogFea.hog_extract();
		//cout << "hog_vector.size: " << hog_vector.size() << endl;

		// 合并到result_vector中
		result_vector.insert(result_vector.end(), color_vector.begin(), color_vector.end());
		result_vector.insert(result_vector.end(), hog_vector.begin(), hog_vector.end());
	}
	return result_vector;
}

/*计算正样本*/
void ClassifierTrain::cal_pos_vector(const string &img_files_dir, const string &label_files_dir, const string &save_file_name)
{
	glob(img_files_dir, img_files, true);     // opencv读取文件夹下所有符合要求的文件路径
	glob(label_files_dir, label_files, true);

	img_numbers = img_files.size();            // 文件个数
	
	out_file.open(save_file_name);                      // 输出文件
	for (int index = 0; index < img_numbers; index++)
	{

		cout << "cal_pos_vector... " << index << endl;

		// 读取图片和解析文件
		cout << "file name: " << img_files[index] << endl;
		src_img = imread(img_files[index]);
		Mat row_img = src_img.clone();        //拷贝方式是深拷贝

		*labels = parse_xml(label_files[index]);
		*new_rect = reshape_ballRect(labels);

		result_vector = get_result_vector(src_img, new_rect);

		for (int i = 0; i < result_vector.size();i++)
			out_file << round(result_vector.at(i), 4) << " ";
		out_file << endl;

		cout << "result_vector.size: " << result_vector.size() << endl;

		/*画出矩形框, 训练时需要注释掉*/
		/*
		for (int i = 0; i < new_rect->size(); i++)
		{
			rectangle(row_img, new_rect->at(i), Scalar(0, 0, 255), 1);
		}
		namedWindow("row_img", 0);
		imshow("row_img", row_img);
		waitKey(100);
		destroyAllWindows();
		*/

		vector<int>().swap(*labels);
		vector<Rect>().swap(*new_rect);
		vector<float>().swap(result_vector);  		      // 清空vector并释放内存
	}
	out_file.close();
}

/*计算负样本*/
void ClassifierTrain::cal_neg_vector(const string &img_files_dir, const string &save_file_name)
{
	glob(img_files_dir, img_files, true);     // opencv读取文件夹下所有符合要求的文件路径

	img_numbers = img_files.size();            // 文件个数

	out_file.open(save_file_name);                        // 输出文件
	for (int index = 0; index < img_numbers; index++)
	{
		cout << "cal_neg_vector... " << index << endl;

		// 读取图片和解析文件
		cout << "file name: " << img_files[index] << endl;
		src_img = imread(img_files[index]);
		Mat row_img = src_img.clone();        //拷贝方式是深拷贝

		/*轮廓检测*/
		ContoursDet con_det(src_img);
		*result_contours = con_det.contours_detecte("football", false, 100, 700, 0, 2);

		cout << "result_contours->size(): " << result_contours->size() << endl;

		for (int index = 0; index < result_contours->size(); index++)
		{
			*labels = contours_to_labels(result_contours->at(index));
			*new_rect = reshape_ballRect(labels);

			result_vector = get_result_vector(src_img, new_rect);

			for (int i = 0; i < result_vector.size(); i++)
				out_file << round(result_vector.at(i), 4) << " ";
			out_file << endl;

			cout << "result_vector.size: " << result_vector.size() << endl;

			/*画出矩形框, 训练时需要注释掉*/
			/*
			for (int i = 0; i < new_rect->size(); i++)
			{
				rectangle(row_img, new_rect->at(i), Scalar(0, 0, 255), 1);
			}
			*/
			
			vector<int>().swap(*labels);
			vector<Rect>().swap(*new_rect);
			vector<float>().swap(result_vector);  		      // 清空vector并释放内存
		}

		/*画出矩形框, 训练时需要注释掉*/
		/*
		namedWindow("row_img", 0);
		imshow("row_img", row_img);
		waitKey(0);
		destroyAllWindows();
		*/
			
	}
	out_file.close();
}

/*测试图片*/
void ClassifierTrain::result_test(const string &img_files_dir)
{
	Classifier myclassifier("data.xml");
	DataSet dataSet;
	Mat result_mat;
	int result;
	dataSet = myclassifier.load_dataset();
	myclassifier.classify_train(dataSet);

	vector<String> img_files;     // 存放文件路径  

	glob(img_files_dir, img_files, true);     // opencv读取文件夹下所有符合要求的文件路径

	size_t img_numbers = img_files.size();            // 文件个数

	for (int index = 0; index < img_numbers; index++)
	{
		cout << "predicting..." << endl;

		src_img = imread(img_files[index]);
		Mat row_img = src_img.clone();        //拷贝方式是深拷贝
		ContoursDet con_det(src_img);
		*result_contours = con_det.contours_detecte("football", false);

		for (int index = 0; index < result_contours->size(); index++)
		{
			*labels = contours_to_labels(result_contours->at(index));
			*new_rect = reshape_ballRect(labels);

			result_vector = get_result_vector(src_img, new_rect);

			cout << "result_vector.size: " << result_vector.size() << endl;

			result_mat = Mat(result_vector, true);
			transpose(result_mat, result_mat);

			result = myclassifier.classify_result(result_mat);
			
			if (result == 1)  rectangle(row_img, result_contours->at(index), Scalar(0, 0, 255), 1);
			else              rectangle(row_img, result_contours->at(index), Scalar(0, 255, 255), 1);
	
			vector<int>().swap(*labels);
			vector<Rect>().swap(*new_rect);
			vector<float>().swap(result_vector);  		      // 清空vector并释放内存

		}

		namedWindow("row_img", 0);
		imshow("row_img", row_img);
		waitKey(0);
		destroyAllWindows();
	}
}
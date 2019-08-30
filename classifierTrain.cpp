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

/*��������������λ��ЧС��*/
double ClassifierTrain::round(double number, int bits)
{
	stringstream ss;
	ss << fixed << setprecision(bits) << number;
	ss >> number;

	return number;
}

/*����xml�ļ�����https://www.sourceforge.net/projects/tinyxml/����Դ���룬����
tinystr.h��tinystr.cpp��tinyxml.h��tinyxml.cpp��tinyxmlerror.cpp��tinyxmlparser.cpp���빤���ļ�����
*/
vector<int> ClassifierTrain::parse_xml(const string &strXmlPath)
{


	//��ȡxml�ļ��еĲ���ֵ
	TiXmlDocument* Document = new TiXmlDocument();
	if (!Document->LoadFile(strXmlPath.c_str()))
	{
		cout << "�޷�����xml�ļ���" << endl;
	}
	TiXmlElement* RootElement = Document->RootElement();		    //��Ŀ¼

	TiXmlElement* NextElement = RootElement->FirstChildElement();  //��Ŀ¼�µĵ�һ���ڵ��

	while (NextElement != NULL)		//�ж���û�ж���
	{
		if (NextElement->ValueTStr() == "object")		//����object�ڵ�
		{
			//NextElement = NextElement->NextSiblingElement();
			TiXmlElement* BoxElement = NextElement->FirstChildElement();
			while (BoxElement->ValueTStr() != "bndbox")		//����box�ڵ�
			{
				BoxElement = BoxElement->NextSiblingElement();
			}
			//������xmin�ڵ�
			TiXmlElement* xminElemeng = BoxElement->FirstChildElement();
			{
				//�ֱ��ȡ�ĸ���ֵ
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

	//�ͷ��ڴ�
	delete Document;
	return *labels;
}

/*��labels��ϢתΪrect��Ϣ*/
vector<Rect> ClassifierTrain::reshape_ballRect(vector<int> *labels)
{
	int x1 = labels->at(0), y1 = labels->at(1), x2 = labels->at(2), y2= labels->at(3);
	int w = int(x2 - x1), h = int(y2 - y1);

	// ��֤w,h��ż����������ǣ���С1�����أ�Ҳ��Ӱ����
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

/*������ϢתΪlabels��Ϣ*/
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
	// ����ÿ��С����
	for (int i = 0; i < new_rect->size(); i++)
	{
		Mat new_rect_area = img(new_rect->at(i));       // �õ�С���ε�����

		// ��ȡ��ɫ����
		ColorFeature colFea(new_rect_area, color_numbers);
		color_vector = colFea.color_extract();
		//cout << "color_extract.size: " << color_vector.size() << endl;

		// ��ȡHOG����
		windowSize = Size(new_rect->at(i).width - 2, new_rect->at(i).height - 2);
		HogFeature hogFea(new_rect_area, windowSize);
		hog_vector = hogFea.hog_extract();
		//cout << "hog_vector.size: " << hog_vector.size() << endl;

		// �ϲ���result_vector��
		result_vector.insert(result_vector.end(), color_vector.begin(), color_vector.end());
		result_vector.insert(result_vector.end(), hog_vector.begin(), hog_vector.end());
	}
	return result_vector;
}

/*����������*/
void ClassifierTrain::cal_pos_vector(const string &img_files_dir, const string &label_files_dir, const string &save_file_name)
{
	glob(img_files_dir, img_files, true);     // opencv��ȡ�ļ��������з���Ҫ����ļ�·��
	glob(label_files_dir, label_files, true);

	img_numbers = img_files.size();            // �ļ�����
	
	out_file.open(save_file_name);                      // ����ļ�
	for (int index = 0; index < img_numbers; index++)
	{

		cout << "cal_pos_vector... " << index << endl;

		// ��ȡͼƬ�ͽ����ļ�
		cout << "file name: " << img_files[index] << endl;
		src_img = imread(img_files[index]);
		Mat row_img = src_img.clone();        //������ʽ�����

		*labels = parse_xml(label_files[index]);
		*new_rect = reshape_ballRect(labels);

		result_vector = get_result_vector(src_img, new_rect);

		for (int i = 0; i < result_vector.size();i++)
			out_file << round(result_vector.at(i), 4) << " ";
		out_file << endl;

		cout << "result_vector.size: " << result_vector.size() << endl;

		/*�������ο�, ѵ��ʱ��Ҫע�͵�*/
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
		vector<float>().swap(result_vector);  		      // ���vector���ͷ��ڴ�
	}
	out_file.close();
}

/*���㸺����*/
void ClassifierTrain::cal_neg_vector(const string &img_files_dir, const string &save_file_name)
{
	glob(img_files_dir, img_files, true);     // opencv��ȡ�ļ��������з���Ҫ����ļ�·��

	img_numbers = img_files.size();            // �ļ�����

	out_file.open(save_file_name);                        // ����ļ�
	for (int index = 0; index < img_numbers; index++)
	{
		cout << "cal_neg_vector... " << index << endl;

		// ��ȡͼƬ�ͽ����ļ�
		cout << "file name: " << img_files[index] << endl;
		src_img = imread(img_files[index]);
		Mat row_img = src_img.clone();        //������ʽ�����

		/*�������*/
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

			/*�������ο�, ѵ��ʱ��Ҫע�͵�*/
			/*
			for (int i = 0; i < new_rect->size(); i++)
			{
				rectangle(row_img, new_rect->at(i), Scalar(0, 0, 255), 1);
			}
			*/
			
			vector<int>().swap(*labels);
			vector<Rect>().swap(*new_rect);
			vector<float>().swap(result_vector);  		      // ���vector���ͷ��ڴ�
		}

		/*�������ο�, ѵ��ʱ��Ҫע�͵�*/
		/*
		namedWindow("row_img", 0);
		imshow("row_img", row_img);
		waitKey(0);
		destroyAllWindows();
		*/
			
	}
	out_file.close();
}

/*����ͼƬ*/
void ClassifierTrain::result_test(const string &img_files_dir)
{
	Classifier myclassifier("data.xml");
	DataSet dataSet;
	Mat result_mat;
	int result;
	dataSet = myclassifier.load_dataset();
	myclassifier.classify_train(dataSet);

	vector<String> img_files;     // ����ļ�·��  

	glob(img_files_dir, img_files, true);     // opencv��ȡ�ļ��������з���Ҫ����ļ�·��

	size_t img_numbers = img_files.size();            // �ļ�����

	for (int index = 0; index < img_numbers; index++)
	{
		cout << "predicting..." << endl;

		src_img = imread(img_files[index]);
		Mat row_img = src_img.clone();        //������ʽ�����
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
			vector<float>().swap(result_vector);  		      // ���vector���ͷ��ڴ�

		}

		namedWindow("row_img", 0);
		imshow("row_img", row_img);
		waitKey(0);
		destroyAllWindows();
	}
}
#include "classifierTrain.h"

int main()
{
	clock_t start, end;
	start = clock();
	ClassifierTrain train;
	//train.cal_pos_vector("img_train_pos/*.jpg", "label_train_pos/*.xml", "data_1.txt");
	//train.cal_neg_vector("img_train_neg/*.jpg", "data_2.txt");
	train.result_test("img_test/*.jpg");
	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Total time:" << endtime << endl;		
	
	return 0;
}

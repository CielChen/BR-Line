/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: �����Ѿ���õ�����+��Ӱ�ָ�ͼ��������Ч����Ӱ����ȡ��������������Ѱ��BR��
------------------------------------------------
*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include "grabShadow.h"
#include "shadowExtraction.h"

using namespace cv;
using namespace std;

int main()
{
	//step1.��ȡ��Ӱ��������BR
	//cutShadow();  //Grabcut�㷨ֻ������Ч����Ӱ
	shadowExtraction();  //��ȡ�����ͱ�������Ч��Ӱ����������Ч��Ӱ����ÿ�����ص�BR

	//step2.�ҵ�BR��С��
	findDot();

	//step3.��С���˷������ֱ��y=kx+b������k��b��ֵ
	drawLine();

	system("pause");
	return 0;
}

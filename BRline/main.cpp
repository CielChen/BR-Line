/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: 读入已经获得的物体+阴影分割图，仅将有效的阴影区提取出来，并在其中寻找BR线
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
	//step1.提取阴影，并计算BR
	//cutShadow();  //Grabcut算法只保留有效的阴影
	shadowExtraction();  //提取场景和背景的有效阴影，并计算有效阴影区中每个像素的BR

	//step2.找到BR最小点
	findDot();

	//step3.最小二乘法，拟合直线y=kx+b，返回k和b的值
	drawLine();

	system("pause");
	return 0;
}

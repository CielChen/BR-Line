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
#include "grabShadow.h"
#include "shadowExtraction.h"

using namespace cv;
using namespace std;

int main()
{
	//step1.提取阴影
	//cutShadow();  //Grabcut算法只保留有效的阴影
	shadowExtraction();  //提取场景和背景的有效阴影

	return 0;
}

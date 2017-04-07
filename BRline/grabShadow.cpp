/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: 读入已经获得的物体+阴影分割图，使用Grabcut法仅将有效的阴影区提取出来
------------------------------------------------
*/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <Windows.h>
#include "grabShadow.h"

using namespace cv;
using namespace std;

Rect selection;
Mat img,img0;
Point prePt(-1,-1);  //二维点(-1,-1)

//判断鼠标动作
void onMouse(int event, int x, int y, int flags, void* param)
{
	if(event == CV_EVENT_LBUTTONDOWN){  //CV_EVENT_LBUTTONDOWN 左键点击
		prePt=Point(x,y);
	}
	else if(event == CV_EVENT_MOUSEMOVE && (flags && CV_EVENT_FLAG_LBUTTON)){  //CV_EVENT_MOUSEMOVE 滑动；CV_EVENT_FLAG_LBUTTON 左键拖拽
		img=img0.clone();
		
		//rectangle()：创建矩形。图像被画到img上，矩形两个对角顶点，矩形颜色，矩形框的粗细
		//Rect（）：矩形左上角的坐标、宽、高
		//Scalar(0, 0, 255):表示RGB颜色，R=255，G=0，B=0
		rectangle(img, Rect(prePt.x, prePt.y, abs(x-prePt.x), abs(y-prePt.y) ), Scalar(0, 0, 255), 3);

		selection=Rect(prePt.x, prePt.y, abs(x-prePt.x), abs(y-prePt.y) );
	}
	else if(event == CV_EVENT_LBUTTONUP || !(flags && CV_EVENT_FLAG_LBUTTON) ){  //CV_EVENT_LBUTTONUP左键放开
		prePt=Point(-1,-1);
	}
	imshow("原图", img);
}

int cutShadow()
{
	//读入场景图片
	img=imread("F:\\Code\\BRline\\Data\\Processed\\20170228111043_brightness+local+spacial.bmp",1);
	
	img0=img.clone();
	namedWindow("原图", WINDOW_AUTOSIZE);
	imshow("原图",img);

	/*setMouseCallback()
	第一个参数：窗口名字
	第二个参数：鼠标相应函数，回调函数。指定窗口里每次鼠标时间发送的时候，被调用的函数指针。*/
	setMouseCallback("原图", onMouse);
	
	Mat result;  //4种可能结果
	Mat bgModel, fgModel;  //背景，前景
	cout<<"--------GrabCut算法----------"<<endl;
	cout<<"---------请手动选择阴影，选好后按p；退出请按q-----"<<endl;
	for(;;){
		int c=waitKey(10);
		if((char)c == 'p'){
			/*grabCut()
			第一个参数：输入图像，类型必须为CV_8UC3
			第二个参数mask：分段结果。大小同第一个参数，但格式为CV_8UC1，只能是单通道的。
			有4种可能结果：背景，前景，可能背景，可能前景
			第三个参数：包含前景的矩阵
			第四个、第五个参数：前景、背景
			第六个参数：迭代次数。迭代次数越多，效果越好，但时间也越长
			第七个参数：用矩形
			*/
			cout<<"正在处理图像···"<<endl;
			grabCut(img0, result, selection, bgModel, fgModel, 5, GC_INIT_WITH_RECT);
			compare(result, GC_PR_FGD, result, CMP_EQ);  //得到前景mask；GC_PR_FGD前景，CMP_EQ比较两个矩阵是否相等

			Mat shadow(img.size(), CV_8UC3, Scalar::all(255));
			img0.copyTo(shadow, result);
			namedWindow("Shadow", WINDOW_AUTOSIZE);
			imshow("Shadow",shadow);
			//保存前景图
			imwrite("F:\\Code\\BRline\\Data\\Shadow\\20170228111043_shadow.bmp",shadow);
		}
		if(char(c)=='q')
			return 0;
	}
	return 0;
}

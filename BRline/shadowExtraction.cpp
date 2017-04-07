/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: 读入场景图和背景图，将有效的阴影区提取出来
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
#include "shadowExtraction.h"

using namespace cv;
using namespace std;

#define WIDTH 1216  //HoloLens视频截图宽度
#define HEIGHT 684  //HoloLens视频截图高度

#define MAX 100

Mat processedMat;  //检测结果图像
Mat sceneShadowMat, backShadowMat;  //存储场景和背景中的有效阴影
Mat BRMat;  //存储X轴方向BR最小点

int processedRGB_B[HEIGHT][WIDTH],processedRGB_G[HEIGHT][WIDTH],processedRGB_R[HEIGHT][WIDTH];  //检测结果图像的RGB分量
int sceneShadowRGB_B[HEIGHT][WIDTH],sceneShadowRGB_G[HEIGHT][WIDTH],sceneShadowRGB_R[HEIGHT][WIDTH];  //前景图像的RGB分量
int backgroundShadowRGB_B[HEIGHT][WIDTH],backgroundShadowRGB_G[HEIGHT][WIDTH],backgroundShadowRGB_R[HEIGHT][WIDTH];  //背景图像的RGB分量
double br_B[HEIGHT][WIDTH],br_G[HEIGHT][WIDTH],br_R[HEIGHT][WIDTH];  //亮度差的RGB分量

double br_minB[WIDTH]={MAX};  //存储x轴方向的BR最小值的y坐标
double br_minG[WIDTH]={MAX};  
double br_minR[WIDTH]={MAX};  

struct pixelInformation{   //结构体存放每个像素点的信息
	int category;  //判断像素点的种类。0，无效像素；1，阴影
//	int initColor_B;  //原图的RBG颜色-B
//	int initColor_G;  //原图的RBG颜色-G
//	int initColor_R;  //原图的RBG颜色-R
//	int revise;  //判断该像素是否被修改。0，没有被修改；1，被修改
};
struct pixelInformation graph[HEIGHT][WIDTH];

int shadowExtraction()
{
	//读入在检测结果图像中提取到的阴影区
	processedMat=imread("F:\\Code\\BRline\\Data\\Shadow\\20170228111043_shadow.bmp");
	namedWindow("shadow in processed picture");
	imshow("shadow in processed picture",processedMat);
	waitKey(0);
	//遍历每个像素
	for(int i=0;i<processedMat.rows;i++)
	{
		const Vec3b* processedPoint=processedMat.ptr<Vec3b>(i);
		for(int j=0;j<processedMat.cols;j++)
		{
			Vec3b intensity=*(processedPoint+j);
			processedRGB_B[i][j]=intensity[0];
			processedRGB_G[i][j]=intensity[1];
			processedRGB_R[i][j]=intensity[2];

			//初始化结构体颜色，颜色同检测结果图像
			if(processedRGB_B[i][j]==0 && processedRGB_G[i][j]==255 && processedRGB_R[i][j]==0)  //阴影
				graph[i][j].category=1;
			else
				graph[i][j].category=0;  //无效像素
		}
	}

	//提取场景中的有效阴影
	sceneShadowMat=imread("F:\\Code\\BRline\\Data\\Scene\\20170228111043_scene.jpg");
	namedWindow("scene picture");
	imshow("scene picture",sceneShadowMat);
	waitKey(0);
	//遍历每个像素
	for(int i=0;i<sceneShadowMat.rows;i++)
	{
		for(int j=0;j<sceneShadowMat.cols;j++)
		{
			if(graph[i][j].category==0)  //无效像素，设置为黑色
			{
				sceneShadowMat.at<Vec3b>(i,j)[0]=0;
				sceneShadowMat.at<Vec3b>(i,j)[1]=0;
				sceneShadowMat.at<Vec3b>(i,j)[2]=0;
			}
		}
	}
	namedWindow("scene shadow picture");
	imshow("scene shadow picture",sceneShadowMat);
	waitKey(0);

	//提取背景中的有效阴影
	backShadowMat=imread("F:\\Code\\BRline\\Data\\Background\\20170228111043_back.jpg");
	namedWindow("background picture");
	imshow("background picture",backShadowMat);
	waitKey(0);
	//遍历每个像素
	for(int i=0;i<backShadowMat.rows;i++)
	{
		for(int j=0;j<backShadowMat.cols;j++)
		{
			if(graph[i][j].category==0)  //无效像素，设置为黑色
			{
				backShadowMat.at<Vec3b>(i,j)[0]=0;
				backShadowMat.at<Vec3b>(i,j)[1]=0;
				backShadowMat.at<Vec3b>(i,j)[2]=0;
			}
		}
	}
	namedWindow("background shadow picture");
	imshow("background shadow picture",backShadowMat);
	waitKey(0);

	//计算阴影区的BR
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			if(graph[i][j].category==0)  //无效像素，BR为MAX
			{
				br_B[i][j]=MAX;
				br_G[i][j]=MAX;
				br_R[i][j]=MAX;
			}
			else
			{
				br_B[i][j]=sceneShadowRGB_B[i][j]/backgroundShadowRGB_B[i][j];
				br_G[i][j]=sceneShadowRGB_G[i][j]/backgroundShadowRGB_G[i][j];
				br_R[i][j]=sceneShadowRGB_R[i][j]/backgroundShadowRGB_R[i][j];
			}
		}
	}

	//----------------找x轴方向BR最小点----------------
	BRMat=sceneShadowMat.clone();
	for(int j=0;j<BRMat.cols;j++)
	{
		for(int i=0;i<BRMat.rows;i++)
		{
			if(graph[i][j].category!=0)  //阴影区
			{
				if(br_minB[j]>br_B[i][j])  //B分量
					br_minB[j]=i;
				if(br_minG[j]>br_G[i][j])  //G分量
					br_minG[j]=i;
				if(br_minR[j]>br_R[i][j])  //R分量
					br_minR[j]=i;
			}
		}
	}
	for(int j=0;j<BRMat.cols;j++)
	{
		for(int i=0;i<BRMat.rows;i++)
		{
			if(graph[i][j].category!=0)  //阴影区
			{
				if(i!=br_minB[j])  //不是最小值，去除
				{
					BRMat.at<Vec3b>(i,j)[0]=0;
					BRMat.at<Vec3b>(i,j)[1]=0;
					BRMat.at<Vec3b>(i,j)[2]=0;
				}
			}
		}
	}
	namedWindow("BR line");
	imshow("BR line",BRMat);
	waitKey(0);

	return 0;
}
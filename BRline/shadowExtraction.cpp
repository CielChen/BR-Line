/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: ���볡��ͼ�ͱ���ͼ������Ч����Ӱ����ȡ����
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

#define WIDTH 1216  //HoloLens��Ƶ��ͼ���
#define HEIGHT 684  //HoloLens��Ƶ��ͼ�߶�

#define MAX 100

Mat processedMat;  //�����ͼ��
Mat sceneShadowMat, backShadowMat;  //�洢�����ͱ����е���Ч��Ӱ
Mat BRMat;  //�洢X�᷽��BR��С��

int processedRGB_B[HEIGHT][WIDTH],processedRGB_G[HEIGHT][WIDTH],processedRGB_R[HEIGHT][WIDTH];  //�����ͼ���RGB����
int sceneShadowRGB_B[HEIGHT][WIDTH],sceneShadowRGB_G[HEIGHT][WIDTH],sceneShadowRGB_R[HEIGHT][WIDTH];  //ǰ��ͼ���RGB����
int backgroundShadowRGB_B[HEIGHT][WIDTH],backgroundShadowRGB_G[HEIGHT][WIDTH],backgroundShadowRGB_R[HEIGHT][WIDTH];  //����ͼ���RGB����
double br_B[HEIGHT][WIDTH],br_G[HEIGHT][WIDTH],br_R[HEIGHT][WIDTH];  //���Ȳ��RGB����

double br_minB[WIDTH]={MAX};  //�洢x�᷽���BR��Сֵ��y����
double br_minG[WIDTH]={MAX};  
double br_minR[WIDTH]={MAX};  

struct pixelInformation{   //�ṹ����ÿ�����ص����Ϣ
	int category;  //�ж����ص�����ࡣ0����Ч���أ�1����Ӱ
//	int initColor_B;  //ԭͼ��RBG��ɫ-B
//	int initColor_G;  //ԭͼ��RBG��ɫ-G
//	int initColor_R;  //ԭͼ��RBG��ɫ-R
//	int revise;  //�жϸ������Ƿ��޸ġ�0��û�б��޸ģ�1�����޸�
};
struct pixelInformation graph[HEIGHT][WIDTH];

int shadowExtraction()
{
	//�����ڼ����ͼ������ȡ������Ӱ��
	processedMat=imread("F:\\Code\\BRline\\Data\\Shadow\\20170228111043_shadow.bmp");
	namedWindow("shadow in processed picture");
	imshow("shadow in processed picture",processedMat);
	waitKey(0);
	//����ÿ������
	for(int i=0;i<processedMat.rows;i++)
	{
		const Vec3b* processedPoint=processedMat.ptr<Vec3b>(i);
		for(int j=0;j<processedMat.cols;j++)
		{
			Vec3b intensity=*(processedPoint+j);
			processedRGB_B[i][j]=intensity[0];
			processedRGB_G[i][j]=intensity[1];
			processedRGB_R[i][j]=intensity[2];

			//��ʼ���ṹ����ɫ����ɫͬ�����ͼ��
			if(processedRGB_B[i][j]==0 && processedRGB_G[i][j]==255 && processedRGB_R[i][j]==0)  //��Ӱ
				graph[i][j].category=1;
			else
				graph[i][j].category=0;  //��Ч����
		}
	}

	//��ȡ�����е���Ч��Ӱ
	sceneShadowMat=imread("F:\\Code\\BRline\\Data\\Scene\\20170228111043_scene.jpg");
	namedWindow("scene picture");
	imshow("scene picture",sceneShadowMat);
	waitKey(0);
	//����ÿ������
	for(int i=0;i<sceneShadowMat.rows;i++)
	{
		for(int j=0;j<sceneShadowMat.cols;j++)
		{
			if(graph[i][j].category==0)  //��Ч���أ�����Ϊ��ɫ
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

	//��ȡ�����е���Ч��Ӱ
	backShadowMat=imread("F:\\Code\\BRline\\Data\\Background\\20170228111043_back.jpg");
	namedWindow("background picture");
	imshow("background picture",backShadowMat);
	waitKey(0);
	//����ÿ������
	for(int i=0;i<backShadowMat.rows;i++)
	{
		for(int j=0;j<backShadowMat.cols;j++)
		{
			if(graph[i][j].category==0)  //��Ч���أ�����Ϊ��ɫ
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

	//������Ӱ����BR
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			if(graph[i][j].category==0)  //��Ч���أ�BRΪMAX
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

	//----------------��x�᷽��BR��С��----------------
	BRMat=sceneShadowMat.clone();
	for(int j=0;j<BRMat.cols;j++)
	{
		for(int i=0;i<BRMat.rows;i++)
		{
			if(graph[i][j].category!=0)  //��Ӱ��
			{
				if(br_minB[j]>br_B[i][j])  //B����
					br_minB[j]=i;
				if(br_minG[j]>br_G[i][j])  //G����
					br_minG[j]=i;
				if(br_minR[j]>br_R[i][j])  //R����
					br_minR[j]=i;
			}
		}
	}
	for(int j=0;j<BRMat.cols;j++)
	{
		for(int i=0;i<BRMat.rows;i++)
		{
			if(graph[i][j].category!=0)  //��Ӱ��
			{
				if(i!=br_minB[j])  //������Сֵ��ȥ��
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
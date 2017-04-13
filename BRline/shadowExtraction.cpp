/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: 
step1.���볡��ͼ�ͱ���ͼ������Ч����Ӱ����ȡ����
step2.������Ч��Ӱ����ÿ�����ص�BR
step3.��x�᷽���ϣ��ҵ�BR��С�����أ�ȥ������Ӱ���ı߽磩
step4.������С���˷�����ϳ�BR��
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
#include <algorithm>
#include <Eigen/Dense>
#include <valarray>
#include "shadowExtraction.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

#define WIDTH 1216  //HoloLens��Ƶ��ͼ���
#define HEIGHT 684  //HoloLens��Ƶ��ͼ�߶�

Mat processedMat;  //�����ͼ��
Mat sceneShadowMat, backShadowMat;  //�洢�����ͱ����е���Ч��Ӱ
//Mat BRMat_B,BRMat_G,BRMat_R;  //�洢X�᷽��BR��С��
Mat BRMat;

int processedRGB_B[HEIGHT][WIDTH],processedRGB_G[HEIGHT][WIDTH],processedRGB_R[HEIGHT][WIDTH];  //�����ͼ���RGB����
int backgroundShadowRGB_B[HEIGHT][WIDTH],backgroundShadowRGB_G[HEIGHT][WIDTH],backgroundShadowRGB_R[HEIGHT][WIDTH];  //����ͼ���RGB����
int backRGB_B[HEIGHT][WIDTH],backRGB_G[HEIGHT][WIDTH],backRGB_R[HEIGHT][WIDTH];
double br_B[HEIGHT][WIDTH],br_G[HEIGHT][WIDTH],br_R[HEIGHT][WIDTH];  //���Ȳ��RGB����

//double br_minB[WIDTH]={INT_MAX};  //�洢x�᷽���BR��Сֵ��y����
//double br_minG[WIDTH]={INT_MAX};  
//double br_minR[WIDTH]={INT_MAX};  
vector<int> line_B;  //��x�᷽��BR_B��Сֵ��(x,y)����
vector<int> line_G;
vector<int> line_R;

vector<Point> dotPoint;  //����BR��Сֵ���X��Y����

struct pixelInformation{   //�ṹ����ÿ�����ص����Ϣ
	int category;  //�ж����ص�����ࡣ0����Ч���أ�1����Ӱ
	int scene_B;  //����ͼBGR
	int scene_G;
	int scene_R;
	int back_B;  //����ͼBGR
	int back_G;
	int back_R;
	double BR_B;  //BR
	double BR_G;
	double BR_R;
	bool border;  //�ж��Ƿ�Ϊ��Ӱ��Ե��true,��Ե��false���Ǳ�Ե
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

	//----------------------------------------��ȡ�����е���Ч��Ӱ----------------------------------------------
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

	//���½ṹ��graph
	for(int i=0;i<sceneShadowMat.rows;i++)
	{
		const Vec3b* sceneShadowPoint=sceneShadowMat.ptr<Vec3b>(i);			
		for(int j=0;j<sceneShadowMat.cols;j++)
		{
			Vec3b intensity=*(sceneShadowPoint+j);
			graph[i][j].scene_B=intensity[0];
			graph[i][j].scene_G=intensity[1];
			graph[i][j].scene_R=intensity[2];
		}
	}

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

	//���½ṹ��graph
	for(int i=0;i<backShadowMat.rows;i++)
	{
		const Vec3b* backShadowPoint=backShadowMat.ptr<Vec3b>(i);			
		for(int j=0;j<backShadowMat.cols;j++)
		{
			Vec3b intensity=*(backShadowPoint+j);
			graph[i][j].back_B=intensity[0];
			graph[i][j].back_G=intensity[1];
			graph[i][j].back_R=intensity[2];
		}
	}

	//������Ӱ����BR
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			if(graph[i][j].category==0)  //��Ч���أ�BRΪMAX
			{
				br_B[i][j]=INT_MAX;
				br_G[i][j]=INT_MAX;
				br_R[i][j]=INT_MAX;
			}
			else
			{
				//ע�����ܻ���ַ�ĸΪ��������������һ��Ҫ�жϣ������������ĸΪ�㣬������Ϊ����С��������
				if(backgroundShadowRGB_B[i][j]==0)
					backgroundShadowRGB_B[i][j]=INT_MIN;
				if(backgroundShadowRGB_G[i][j]==0)
					backgroundShadowRGB_G[i][j]=INT_MIN;
				if(backgroundShadowRGB_R[i][j]==0)
					backgroundShadowRGB_R[i][j]=INT_MIN;

				br_B[i][j]=(double)graph[i][j].scene_B/backgroundShadowRGB_B[i][j];
				br_G[i][j]=(double)graph[i][j].scene_G/backgroundShadowRGB_G[i][j];
				br_R[i][j]=(double)graph[i][j].scene_R/backgroundShadowRGB_R[i][j];

				//cout<<br_B[i][j]<<"\t"<<br_G[i][j]<<"\t"<<br_R[i][j]<<endl;
			}
		}
	}

	//���½ṹ��graph
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			graph[i][j].BR_B=br_B[i][j];
			graph[i][j].BR_G=br_G[i][j];
			graph[i][j].BR_R=br_R[i][j];
		}
	}


	//**********************************�鿴graph******************************************
	ofstream graphInfoAll("F:\\Code\\BRline\\Data\\GraphInformation\\graphAll.txt");  //���ļ�
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			graphInfoAll<<graph[i][j].category<<","<<graph[i][j].scene_B<<"/"<<graph[i][j].scene_G<<"/"<<graph[i][j].scene_R<<","<<graph[i][j].back_B<<"/"<<graph[i][j].back_G<<"/"<<graph[i][j].back_R<<","<<graph[i][j].BR_B<<"/"<<graph[i][j].BR_G<<"/"<<graph[i][j].BR_R;
		}
		graphInfoAll<<endl;   //ÿ�������������ӻ���
	}
	graphInfoAll.close();

	//--------------------------ȥ����Ӱ���ı�Ե���������������Ǳ�Ե��Ӱ��BR-----------------------------
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			if(graph[i][j].category==1)
			{
				if(graph[i-1][j].category==1 && graph[i][j-1].category==1 && graph[i][j+1].category==1 && graph[i+1][j].category==1)
				{
					graph[i][j].border=false;
					graph[i][j].BR_B=(graph[i][j].BR_B+graph[i-1][j].BR_B+graph[i][j-1].BR_B+graph[i][j+1].BR_B+graph[i+1][j].BR_B)/5;
					graph[i][j].BR_G=(graph[i][j].BR_G+graph[i-1][j].BR_G+graph[i][j-1].BR_G+graph[i][j+1].BR_G+graph[i+1][j].BR_G)/5;
					graph[i][j].BR_R=(graph[i][j].BR_R+graph[i-1][j].BR_R+graph[i][j-1].BR_R+graph[i][j+1].BR_R+graph[i+1][j].BR_R)/5;
				}
				else
					graph[i][j].border=true;
			}
		}
	}

	return 0;
}

int findMin(int a,int b,int c)//����С
{
	int result;
	if(a<b)
		result=a;
	else
		result=b;
	if(result>c)
		result=c;
	return result;
}


//�ҵ�BR��Сֵ��:pixel[i][j]��i���䣬��j��С
int findDot()
{
	//----------------��x�᷽��BR��С��----------------
	BRMat=sceneShadowMat.clone();
	for(int i=0;i<BRMat.rows;i++)
	{
		int brb=0, brg=0,brr=0;
		int flagB=0,flagG=0,flagR=0;
		for(int j=1;j<BRMat.cols;j++)
		{
			if(graph[i][j].category==1 && graph[i][j].border==false)
			{
				//BR_B
				if(graph[i][j].BR_B<=graph[i][brb].BR_B)
				{
					brb=j;
					flagB=1;
				}
				//BR_G
				if(graph[i][j].BR_G<=graph[i][brg].BR_G)
				{
					brg=j;
					flagG=1;
				}
				//BR_R
				if(graph[i][j].BR_R<=graph[i][brr].BR_R)
				{
					brr=j;
					flagR=1;
				}
			}
		}
		if(flagB==1)
		{
			line_B.push_back(i);
			line_B.push_back(brb);
		}
		if(flagG==1)
		{
			line_G.push_back(i);
			line_G.push_back(brg);
		}
		if(flagR==1)
		{
			line_R.push_back(i);
			line_R.push_back(brr);
		}
	}
	//��ʼ��BRMat��ȫ��
	for(int i=0;i<BRMat.rows;i++)
	{
		for(int j=0;j<BRMat.cols;j++)
		{
			BRMat.at<Vec3b>(i,j)[0]=0;
			BRMat.at<Vec3b>(i,j)[1]=0;
			BRMat.at<Vec3b>(i,j)[2]=0;
		}
	}

	//------------------------�ָ���-----------------------------
	//�ҵ�BR��С�㣬����ͼ�б�ǳ���
	int coordinateX_B,coordinateY_B;
	int coordinateX_G,coordinateY_G;
	int coordinateX_R,coordinateY_R;
	int coordinateX,coordinateY;
	int i=0,j=0,k=0;
	Mat cdst=BRMat.clone();
	for( ; ; )
	{
		if(i==line_B.size() || j==line_G.size() || k==line_R.size())
			break;
		coordinateX_B=line_B.at(i);
		coordinateY_B=line_B.at(i+1);
		coordinateX_G=line_G.at(j);
		coordinateY_G=line_G.at(j+1);
		coordinateX_R=line_R.at(k);
		coordinateY_R=line_R.at(k+1);

		if(coordinateX_B==coordinateX_G && coordinateX_G==coordinateX_R)
		{
			coordinateX=coordinateX_B;
			coordinateY=findMin(coordinateY_B, coordinateY_G, coordinateY_R);

			//ע�⣺���ﱣ������ʱ��X��Y�Ƿ������ģ���������
			dotPoint.push_back(Point(coordinateY,coordinateX));  //��������
			
			/*cout<<"coordinateX="<<coordinateX<<",coordinateY="<<coordinateY<<endl;
			Point dot;  //��������
			dot.x=coordinateY;
			dot.y=coordinateX;
			dotPoint.push_back(dot); 
			cout<<"dot.x="<<dot.x<<",dot.y="<<dot.y<<endl;

			Point center=Point(dot.x, dot.y);
			circle(cdst, center, 1, cv::Scalar(255,0,255));
			*/

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
			
			i=i+2;
			j=j+2;
			k=k+2;
		}
		if(coordinateX_B==coordinateX_G && coordinateX_G!=coordinateX_R)
		{
			if(coordinateX_B<coordinateX_R)
			{
				if(coordinateY_B<=coordinateY_G)
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_B;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������
					
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				}
				else
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_G;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				}
				i=i+2;
				j=j+2;
			}
			else
			{
				k=k+2;
			}
		}
		if(coordinateX_B==coordinateX_R && coordinateX_B!=coordinateX_G)
		{
			if(coordinateX_B<coordinateX_G)
			{
				if(coordinateY_B<=coordinateY_R)
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_B;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				i=i+2;
				k=k+2;
			}
			else
			{
				j=j+2;
			}
		}
		if(coordinateX_G==coordinateX_R && coordinateX_G!=coordinateX_B)
		{
			if(coordinateX_G<coordinateX_B)
			{
				if(coordinateY_G<=coordinateY_R)
				{
					coordinateX=coordinateX_G;
					coordinateY=coordinateY_G;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_G;
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				j=j+2;
				k=k+2;
			}
			else
			{
				i=i+2;
			}
		}
		if(coordinateX_B!=coordinateX_G && coordinateX_B!=coordinateX_R && coordinateX_G!=coordinateX_R)
		{
			int temp;
			temp=findMin(coordinateX_B, coordinateX_G, coordinateX_R);
			coordinateX=temp;
			
			if(temp==coordinateX_B)
			{
				coordinateY=coordinateY_B;
				dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				i=i+2;
			}
			else if(temp==coordinateX_G)
			{
				coordinateY=coordinateY_G;
				dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				j=j+2;
			}
			else
			{
				coordinateY=coordinateY_R;
				dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				k=k+2;
			}
		}
	}
	//����ѭ�������i��j��k������û�дﵽ����ĩβ
	if(i<line_B.size() && j<line_G.size() && k==line_R.size())
	{
		for(; i<line_B.size(), j<line_G.size(); )
		{
			coordinateX_B=line_B.at(i);
			coordinateY_B=line_B.at(i+1);
			coordinateX_G=line_G.at(j);
			coordinateY_G=line_G.at(j+1);
			if(coordinateX_B==coordinateX_G)
			{
				coordinateX=coordinateX_B;
				if(coordinateY_B<=coordinateY_G)
				{			
					coordinateY=coordinateY_B;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;			
				}
				else
				{
					coordinateY=coordinateY_G;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;	
				}
				i=i+2;
				j=j+2;
			}
			else if(coordinateX_B<coordinateX_G)
			{
				i=i+2;
			}
			else
			{
				j=j+2;
			}
		}
	}
	if(i<line_B.size() && k<line_R.size() && j==line_G.size())
	{
		for(; i<line_B.size(), k<line_R.size(); )
		{
			coordinateX_B=line_B.at(i);
			coordinateY_B=line_B.at(i+1);
			coordinateX_R=line_R.at(k);
			coordinateY_R=line_R.at(k+1);
			if(coordinateX_B==coordinateX_R)
			{
				coordinateX=coordinateX_B;
				if(coordinateY_B<=coordinateY_R)
				{			
					coordinateY=coordinateY_B;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;	
				}
				i=i+2;
				k=k+2;
			}
			else if(coordinateX_B<coordinateX_R)
			{
				i=i+2;
			}
			else
			{
				k=k+2;
			}
		}
	}
	if(j<line_G.size() && k<line_R.size() && i==line_B.size())
	{
		for(; j<line_G.size(), k<line_R.size(); )
		{
			coordinateX_G=line_G.at(j);
			coordinateY_G=line_G.at(j+1);
			coordinateX_R=line_R.at(k);
			coordinateY_R=line_R.at(k+1);
			if(coordinateX_G==coordinateX_R)
			{
				coordinateX=coordinateX_G;
				if(coordinateY_G<=coordinateY_R)
				{			
					coordinateY=coordinateY_G;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;	
				}
				j=j+2;
				k=k+2;
			}
			else if(coordinateX_G<coordinateX_R)
			{
				j=j+2;
			}
			else
			{
				k=k+2;
			}
		}
	}
	//����ѭ�������i��j��k��һ��û�дﵽ����ĩβ
	if(i<line_B.size() && j==line_G.size() && k==line_R.size() )
	{
		for(;i<line_B.size();i=i+2)
		{
			coordinateX=line_B.at(i);
			coordinateY=line_B.at(i+1);
			dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
		}
	}
	if(j<line_G.size() && i==line_B.size() && k==line_R.size() )
	{
		for(;j<line_G.size();j=j+2)
		{
			coordinateX=line_G.at(j);
			coordinateY=line_G.at(j+1);
			dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
		}
	}
	if(k<line_R.size() && i==line_B.size() && j==line_G.size() )
	{
		for(;k<line_R.size();k=k+2)
		{
			coordinateX=line_R.at(k);
			coordinateY=line_R.at(k+1);
			dotPoint.push_back(Point(coordinateY,coordinateX));  //��������

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
		}
	}

	namedWindow("BRDot",WINDOW_NORMAL);
	imshow("BRDot", BRMat);
	waitKey(0);
	imwrite("F:\\Code\\BRLine\\Data\\BRDot\\201702281110043_BRdot.bmp", BRMat);

	return 0;
}


/*
//�ҵ�BR��Сֵ��:pixel[i][j]��j���䣬��i��С
int findDot()
{
	//----------------��x�᷽��BR��С��----------------
	BRMat=sceneShadowMat.clone();
	for(int j=0;j<BRMat.cols;j++)
	{
		int brb=0, brg=0,brr=0;
		int flagB=0,flagG=0,flagR=0;
		for(int i=1;i<BRMat.rows;i++)
		{
			if(graph[i][j].category==1)
			{
				//BR_B
				if(graph[i][j].BR_B<graph[brb][j].BR_B)
				{
					brb=i;
					flagB=1;
				}
				//BR_G
				if(graph[i][j].BR_G<graph[brg][j].BR_G)
				{
					brg=i;
					flagG=1;
				}
				//BR_R
				if(graph[i][j].BR_R<graph[brr][j].BR_R)
				{
					brr=i;
					flagR=1;
				}
			}
		}
		if(flagB==1)
		{
			line_B.push_back(brb);
			line_B.push_back(j);
		}
		if(flagG==1)
		{
			line_G.push_back(brg);
			line_G.push_back(j);
		}
		if(flagR==1)
		{
			line_R.push_back(brr);
			line_R.push_back(j);
		}
	}
	//��ʼ��BRMat��ȫ��
	for(int i=0;i<BRMat.rows;i++)
	{
		for(int j=0;j<BRMat.cols;j++)
		{
			BRMat.at<Vec3b>(i,j)[0]=0;
			BRMat.at<Vec3b>(i,j)[1]=0;
			BRMat.at<Vec3b>(i,j)[2]=0;
		}
	}
	//�ҵ�BR��С�㣬����ͼ�б�ǳ���
/*	int coordinateX,coordinateY;
	//BR_B����ɫ
	for(int i=0;i<line_B.size();)
	{
		coordinateX=line_B.at(i);
		coordinateY=line_B.at(i+1);
		i=i+2;

		BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
		BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
		BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
	}
	//BR_G����ɫ
	for(int i=0;i<line_G.size();)
	{
		coordinateX=line_G.at(i);
		coordinateY=line_G.at(i+1);
		i=i+2;

		BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
		BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
		BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
	}
	//BR_R����ɫ
	for(int i=0;i<line_R.size();)
	{
		coordinateX=line_R.at(i);
		coordinateY=line_R.at(i+1);
		i=i+2;

		BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
		BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
		BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
	}
	*/

	/*
	//------------------------�ָ���-----------------------------
	int coordinateX_B,coordinateY_B;
	int coordinateX_G,coordinateY_G;
	int coordinateX_R,coordinateY_R;
	int coordinateX,coordinateY;
	int i=0,j=0,k=0;
	for( ; ; )
	{
		if(i==line_B.size() || j==line_G.size() || k==line_R.size())
			break;
		coordinateX_B=line_B.at(i);
		coordinateY_B=line_B.at(i+1);
		coordinateX_G=line_G.at(j);
		coordinateY_G=line_G.at(j+1);
		coordinateX_R=line_R.at(k);
		coordinateY_R=line_R.at(k+1);

		if(coordinateY_B==coordinateY_G && coordinateY_G==coordinateY_R)
		{
			coordinateY=coordinateY_B;
			coordinateX=findMin(coordinateX_B, coordinateX_G, coordinateX_R);
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
			i=i+2;
			j=j+2;
			k=k+2;
		}
		if(coordinateY_B==coordinateY_G && coordinateY_G!=coordinateY_R)
		{
			if(coordinateY_B<coordinateY_R)
			{
				if(coordinateX_B<=coordinateX_G)
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				}
				else
				{
					coordinateX=coordinateX_G;
					coordinateY=coordinateY_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				}
				i=i+2;
				j=j+2;
			}
			else
			{
				k=k+2;
			}
		}
		if(coordinateY_B==coordinateY_R && coordinateY_B!=coordinateY_G)
		{
			if(coordinateY_B<coordinateY_G)
			{
				if(coordinateX_B<=coordinateX_R)
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_R;
					coordinateY=coordinateY_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				i=i+2;
				k=k+2;
			}
			else
			{
				j=j+2;
			}
		}
		if(coordinateY_G==coordinateY_R && coordinateY_G!=coordinateY_B)
		{
			if(coordinateY_G<coordinateY_B)
			{
				if(coordinateX_G<=coordinateX_R)
				{
					coordinateX=coordinateX_G;
					coordinateY=coordinateY_G;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_R;
					coordinateY=coordinateY_R;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				j=j+2;
				k=k+2;
			}
			else
			{
				i=i+2;
			}
		}
		if(coordinateY_B!=coordinateY_G && coordinateY_B!=coordinateY_R && coordinateY_G!=coordinateY_R)
		{
			int temp;
			temp=findMin(coordinateY_B, coordinateY_G, coordinateY_R);
			coordinateY=temp;
			
			if(temp==coordinateY_B)
			{
				coordinateX=coordinateX_B;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				i=i+2;
			}
			else if(temp==coordinateY_G)
			{
				coordinateX=coordinateX_G;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				j=j+2;
			}
			else
			{
				coordinateX=coordinateX_R;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				k=k+2;
			}
		}
	}
	//����ѭ�������i��j��k������û�дﵽ����ĩβ
	if(i<line_B.size() && j<line_G.size() && k==line_R.size())
	{
		for(; i<line_B.size(), j<line_G.size(); )
		{
			coordinateX_B=line_B.at(i);
			coordinateY_B=line_B.at(i+1);
			coordinateX_G=line_G.at(j);
			coordinateY_G=line_G.at(j+1);
			if(coordinateY_B==coordinateY_G)
			{
				coordinateY=coordinateY_B;
				if(coordinateX_B<=coordinateX_G)
				{			
					coordinateX=coordinateX_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;			
				}
				else
				{
					coordinateX=coordinateX_G;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;	
				}
				i=i+2;
				j=j+2;
			}
			else if(coordinateY_B<coordinateY_G)
			{
				i=i+2;
			}
			else
			{
				j=j+2;
			}
		}
	}
	if(i<line_B.size() && k<line_R.size() && j==line_G.size())
	{
		for(; i<line_B.size(), k<line_R.size(); )
		{
			coordinateX_B=line_B.at(i);
			coordinateY_B=line_B.at(i+1);
			coordinateX_R=line_R.at(k);
			coordinateY_R=line_R.at(k+1);
			if(coordinateY_B==coordinateY_R)
			{
				coordinateY=coordinateY_B;
				if(coordinateX_B<=coordinateX_R)
				{			
					coordinateX=coordinateX_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateX=coordinateX_R;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;	
				}
				i=i+2;
				k=k+2;
			}
			else if(coordinateY_B<coordinateY_R)
			{
				i=i+2;
			}
			else
			{
				k=k+2;
			}
		}
	}
	if(j<line_G.size() && k<line_R.size() && i==line_B.size())
	{
		for(; j<line_G.size(), k<line_R.size(); )
		{
			coordinateX_G=line_G.at(j);
			coordinateY_G=line_G.at(j+1);
			coordinateX_R=line_R.at(k);
			coordinateY_R=line_R.at(k+1);
			if(coordinateY_G==coordinateY_R)
			{
				coordinateY=coordinateY_G;
				if(coordinateX_G<=coordinateX_R)
				{			
					coordinateX=coordinateX_G;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateX=coordinateX_R;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;	
				}
				j=j+2;
				k=k+2;
			}
			else if(coordinateY_G<coordinateY_R)
			{
				j=j+2;
			}
			else
			{
				k=k+2;
			}
		}
	}
	//����ѭ�������i��j��k��һ��û�дﵽ����ĩβ
	if(i<line_B.size() && j==line_G.size() && k==line_R.size() )
	{
		for(;i<line_B.size();i=i+2)
		{
			coordinateX=line_B.at(i);
			coordinateY=line_B.at(i+1);
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
		}
	}
	if(j<line_G.size() && i==line_B.size() && k==line_R.size() )
	{
		for(;j<line_G.size();j=j+2)
		{
			coordinateX=line_G.at(j);
			coordinateY=line_G.at(j+1);
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
		}
	}
	if(k<line_R.size() && i==line_B.size() && j==line_G.size() )
	{
		for(;k<line_R.size();k=k+2)
		{
			coordinateX=line_R.at(k);
			coordinateY=line_R.at(k+1);
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //��ɫ
			BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
			BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
		}
	}
	namedWindow("BRline",WINDOW_NORMAL);
	imshow("BRline", BRMat);
	waitKey(0);
	//imwrite("F:\\Code\\BRline\\Data\\BRLine\\201702281110043_BRline.bmp", BRMat);


	return 0;
}
*/

//��С���˷���������ֱ�ߵ�k,b��y=kx+b��
std::vector<double> leastSquareFitting(std::vector<Point> &rPoints)
{
	std::vector<double> resLine(2);  //����k��b
	int num_points=rPoints.size();
	//std::valarray�Ǳ�ʾ�Ͳ�����ֵ������ࡣ��֧��Ԫ�ؼ������ѧ����͸�����ʽ�Ĺ����±������
	std::valarray<float> data_x(num_points);
	std::valarray<float> data_y(num_points);
	for(int i=0;i<num_points; i++)
	{
		data_x[i]=rPoints[i].x;
		data_y[i]=rPoints[i].y;
	}

	float A=0.0;
	float B=0.0;
	float C=0.0;
	float D=0.0;
	//������С���˷��Ĺ�ʽ��������á����4.12ֽ�ʱʼ�
	A=(data_x*data_x).sum();
	B=data_x.sum();
	C=(data_x*data_y).sum();
	D=data_y.sum();

	float k,b,temp=0;
	if(temp=(A*data_x.size()-B*B))  //��ĸ��Ϊ0
	{
		k=(C*data_x.size()-B*D)/temp;
		b=(A*D-C*B)/temp;
	}
	else
	{
		k=1;
		b=0;
	}
	resLine[0]=k;
	resLine[1]=b;

	return resLine;
}

//������ϵ�ֱ��
//��һ���������õ���BR��Сֵ�㣻�ڶ�����������һ�����ϵĵ㣻��������������������֮��ĵ�
int regressionPoint(const std::vector<Point> &rPoints, std::vector<Point> &rInlinePoints, std::vector<Point> &rOutlinePoints)
{
	int length=rPoints.size();
	std::vector<bool> is_inline;
	for(int i=0;i<length;i++)
		is_inline.push_back(true);

	double avg=0.0;
	std::vector<double> kb;
	bool find_new_points=false;
	while(!find_new_points)  //���ڵ�
	{
		std::vector<int> InlinePointsIndex;
		rInlinePoints.clear();
		for(int i=0;i<length;++i)
		{
			if(is_inline[i])  //if(1):ִ��
			{
				rInlinePoints.push_back(rPoints[i]);
				InlinePointsIndex.push_back(i);
			}
		}

		//�����y=kx+b������k��b
		kb=leastSquareFitting(rInlinePoints);

		//�в�
		std::vector<double> def;
		for(int i=0;i<rInlinePoints.size();++i)
		{
			def.push_back(abs(kb[0]*rInlinePoints[i].x+kb[1]-rInlinePoints[i].y));
			avg+=def[i];
		}

		//�ж��Ƿ����µ�����ĵ�
		avg=avg/rInlinePoints.size();
		find_new_points=false;
		for(int i=0; i<rInlinePoints.size(); ++i)
		{
			if(2.0*avg<def[i])
			{
				is_inline[InlinePointsIndex[i]]=false;
				find_new_points=true;
			}
		}
	}

	//�����
	rOutlinePoints.clear();
	for(int i=0;i<length;++i)
	{
		if(!is_inline[i])  //if��Ϊ0��ִ��
			rOutlinePoints.push_back(rPoints[i]);
	}

	//-------------------------------��ͼ------------------------------
	//��ͼ
	//cv::Mat cdst=cv::Mat::zeros(WIDTH,HEIGHT,CV_8SC3);
	Mat cdst=BRMat.clone();
	//��ͼ���ڵ�:��ɫ
	for(int i=0;i<rInlinePoints.size();i++)
	{
		cv::Point center=cv::Point((int)rInlinePoints[i].x, (int)rInlinePoints[i].y);
		//��Բ
		//��һ��������Ҫ����Բ���ڵ�ͼ��
		//�ڶ���������Բ������
		//�������������뾶
		//���ĸ���������ɫ
		//������������������������ʾ���Բ�������Ĵ�ϸ�̶ȡ����򣬱�ʾԲ�Ƿ����
		circle(cdst,center,1,cv::Scalar(0,255,255),-1);
	}

	//��������㣺��ɫ
	for(int i=0;i<rOutlinePoints.size();i++)
	{
		cv::Point center=cv::Point((int)rOutlinePoints[i].x, (int)rOutlinePoints[i].y);
		circle(cdst, center, 3, cv::Scalar(255,0,255), -1);
	}
	//����:��ɫ
	cv::Point pt1,pt2;
	int beginDot=dotPoint[0].x, endDot=dotPoint[dotPoint.size()-1].x;
	pt1.x=beginDot;
	pt1.y=kb[0]*pt1.x+kb[1];
	pt2.x=endDot;
	pt2.y=kb[0]*pt2.x+kb[1];
	//��һ��������Ҫ���������ڵ�ͼ��
	//�ڶ���������ֱ�����
	//������������ֱ���յ�
	//���ĸ���������ɫ
	//�����������������ϸ
	//������������line_type=8����8�ڽ�)���� ��
	line(cdst, pt1, pt2, cv::Scalar(255,255,255), 1, 8);

	imshow("BRline", cdst);
	cv::namedWindow("BRline",WINDOW_NORMAL);
	cv::waitKey(0);
	imwrite("F:\\Code\\BRLine\\Data\\BRLine\\201702281110043_BRLine.bmp", cdst);

	//������ڵ㡢����㡢�ߵ�kbֵ  
	cout<<"���ڵ㣺"<<endl;
    for(auto w:rInlinePoints) 
		std::cout<<w.x<<","<<w.y<<std::endl;  
    std::cout<<std::endl;
	cout<<"����㣺"<<endl;
    for(auto w:rOutlinePoints) 
		std::cout<<w.x<<","<<w.y<<std::endl;  
    std::cout<<std::endl;  
	cout<<"k��b��"<<endl;
    std::cout<<kb[0]<<","<<kb[1]<<std::endl;  

	return 0;
}

int drawLine()
{
	vector<Point> in_line_points, rOutlinePoints;
	regressionPoint(dotPoint, in_line_points, rOutlinePoints);

	return 0;
}
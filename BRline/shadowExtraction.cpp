/*
------------------------------------------------
Author: CIEL
Date: 2017/04/07
Function: 
step1.读入场景图和背景图，将有效的阴影区提取出来
step2.计算有效阴影区中每个像素的BR
step3.在x轴方向上，找到BR最小的像素（去掉了阴影区的边界）
step4.利用最小二乘法，拟合出BR线
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

#define WIDTH 1216  //HoloLens视频截图宽度
#define HEIGHT 684  //HoloLens视频截图高度

Mat processedMat;  //检测结果图像
Mat sceneShadowMat, backShadowMat;  //存储场景和背景中的有效阴影
//Mat BRMat_B,BRMat_G,BRMat_R;  //存储X轴方向BR最小点
Mat BRMat;

int processedRGB_B[HEIGHT][WIDTH],processedRGB_G[HEIGHT][WIDTH],processedRGB_R[HEIGHT][WIDTH];  //检测结果图像的RGB分量
int backgroundShadowRGB_B[HEIGHT][WIDTH],backgroundShadowRGB_G[HEIGHT][WIDTH],backgroundShadowRGB_R[HEIGHT][WIDTH];  //背景图像的RGB分量
int backRGB_B[HEIGHT][WIDTH],backRGB_G[HEIGHT][WIDTH],backRGB_R[HEIGHT][WIDTH];
double br_B[HEIGHT][WIDTH],br_G[HEIGHT][WIDTH],br_R[HEIGHT][WIDTH];  //亮度差的RGB分量

//double br_minB[WIDTH]={INT_MAX};  //存储x轴方向的BR最小值的y坐标
//double br_minG[WIDTH]={INT_MAX};  
//double br_minR[WIDTH]={INT_MAX};  
vector<int> line_B;  //存x轴方向BR_B最小值的(x,y)坐标
vector<int> line_G;
vector<int> line_R;

vector<Point> dotPoint;  //保存BR最小值点的X和Y坐标

struct pixelInformation{   //结构体存放每个像素点的信息
	int category;  //判断像素点的种类。0，无效像素；1，阴影
	int scene_B;  //场景图BGR
	int scene_G;
	int scene_R;
	int back_B;  //背景图BGR
	int back_G;
	int back_R;
	double BR_B;  //BR
	double BR_G;
	double BR_R;
	bool border;  //判断是否为阴影边缘：true,边缘；false，非边缘
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

	//----------------------------------------提取场景中的有效阴影----------------------------------------------
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

	//更新结构体graph
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

	//更新结构体graph
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

	//计算阴影区的BR
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			if(graph[i][j].category==0)  //无效像素，BR为MAX
			{
				br_B[i][j]=INT_MAX;
				br_G[i][j]=INT_MAX;
				br_R[i][j]=INT_MAX;
			}
			else
			{
				//注：可能会出现分母为零的情况！！！！一定要判断！！！！如果分母为零，则将其设为无穷小！！！！
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

	//更新结构体graph
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			graph[i][j].BR_B=br_B[i][j];
			graph[i][j].BR_G=br_G[i][j];
			graph[i][j].BR_R=br_R[i][j];
		}
	}


	//**********************************查看graph******************************************
	ofstream graphInfoAll("F:\\Code\\BRline\\Data\\GraphInformation\\graphAll.txt");  //打开文件
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			graphInfoAll<<graph[i][j].category<<","<<graph[i][j].scene_B<<"/"<<graph[i][j].scene_G<<"/"<<graph[i][j].scene_R<<","<<graph[i][j].back_B<<"/"<<graph[i][j].back_G<<"/"<<graph[i][j].back_R<<","<<graph[i][j].BR_B<<"/"<<graph[i][j].BR_G<<"/"<<graph[i][j].BR_R;
		}
		graphInfoAll<<endl;   //每行输出结束，添加换行
	}
	graphInfoAll.close();

	//--------------------------去掉阴影区的边缘，并利用邻域计算非边缘阴影的BR-----------------------------
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

int findMin(int a,int b,int c)//找最小
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


//找到BR最小值点:pixel[i][j]，i不变，找j最小
int findDot()
{
	//----------------找x轴方向BR最小点----------------
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
	//初始化BRMat，全黑
	for(int i=0;i<BRMat.rows;i++)
	{
		for(int j=0;j<BRMat.cols;j++)
		{
			BRMat.at<Vec3b>(i,j)[0]=0;
			BRMat.at<Vec3b>(i,j)[1]=0;
			BRMat.at<Vec3b>(i,j)[2]=0;
		}
	}

	//------------------------分割线-----------------------------
	//找到BR最小点，并在图中标记出来
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

			//注意：这里保存坐标时，X和Y是反过来的！！！！！
			dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标
			
			/*cout<<"coordinateX="<<coordinateX<<",coordinateY="<<coordinateY<<endl;
			Point dot;  //保存坐标
			dot.x=coordinateY;
			dot.y=coordinateX;
			dotPoint.push_back(dot); 
			cout<<"dot.x="<<dot.x<<",dot.y="<<dot.y<<endl;

			Point center=Point(dot.x, dot.y);
			circle(cdst, center, 1, cv::Scalar(255,0,255));
			*/

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //白色
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
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标
					
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				}
				else
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_G;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
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
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_B;
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
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
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_G;
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
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
				dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //蓝色
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				i=i+2;
			}
			else if(temp==coordinateX_G)
			{
				coordinateY=coordinateY_G;
				dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //绿色
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				j=j+2;
			}
			else
			{
				coordinateY=coordinateY_R;
				dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //红色
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				k=k+2;
			}
		}
	}
	//跳出循环，如果i，j，k有两个没有达到容器末尾
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
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;			
				}
				else
				{
					coordinateY=coordinateY_G;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
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
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
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
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateY=coordinateY_R;
					dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
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
	//跳出循环，如果i，j，k有一个没有达到容器末尾
	if(i<line_B.size() && j==line_G.size() && k==line_R.size() )
	{
		for(;i<line_B.size();i=i+2)
		{
			coordinateX=line_B.at(i);
			coordinateY=line_B.at(i+1);
			dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //蓝色
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
			dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //绿色
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
			dotPoint.push_back(Point(coordinateY,coordinateX));  //保存坐标

			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //红色
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
//找到BR最小值点:pixel[i][j]，j不变，找i最小
int findDot()
{
	//----------------找x轴方向BR最小点----------------
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
	//初始化BRMat，全黑
	for(int i=0;i<BRMat.rows;i++)
	{
		for(int j=0;j<BRMat.cols;j++)
		{
			BRMat.at<Vec3b>(i,j)[0]=0;
			BRMat.at<Vec3b>(i,j)[1]=0;
			BRMat.at<Vec3b>(i,j)[2]=0;
		}
	}
	//找到BR最小点，并在图中标记出来
/*	int coordinateX,coordinateY;
	//BR_B：蓝色
	for(int i=0;i<line_B.size();)
	{
		coordinateX=line_B.at(i);
		coordinateY=line_B.at(i+1);
		i=i+2;

		BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //蓝色
		BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
		BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
	}
	//BR_G：绿色
	for(int i=0;i<line_G.size();)
	{
		coordinateX=line_G.at(i);
		coordinateY=line_G.at(i+1);
		i=i+2;

		BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //绿色
		BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
		BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
	}
	//BR_R：红色
	for(int i=0;i<line_R.size();)
	{
		coordinateX=line_R.at(i);
		coordinateY=line_R.at(i+1);
		i=i+2;

		BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //红色
		BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
		BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
	}
	*/

	/*
	//------------------------分割线-----------------------------
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
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //白色
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
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				}
				else
				{
					coordinateX=coordinateX_G;
					coordinateY=coordinateY_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
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
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_R;
					coordinateY=coordinateY_B;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
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
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				}
				else
				{
					coordinateX=coordinateX_R;
					coordinateY=coordinateY_R;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
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
				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //蓝色
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				i=i+2;
			}
			else if(temp==coordinateY_G)
			{
				coordinateX=coordinateX_G;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //绿色
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;
				j=j+2;
			}
			else
			{
				coordinateX=coordinateX_R;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //红色
				BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
				BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;
				k=k+2;
			}
		}
	}
	//跳出循环，如果i，j，k有两个没有达到容器末尾
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
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=0;			
				}
				else
				{
					coordinateX=coordinateX_G;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //青色
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
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=0;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateX=coordinateX_R;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //粉色
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
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
					BRMat.at<Vec3b>(coordinateX,coordinateY)[1]=255;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[2]=255;			
				}
				else
				{
					coordinateX=coordinateX_R;
					BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //黄色
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
	//跳出循环，如果i，j，k有一个没有达到容器末尾
	if(i<line_B.size() && j==line_G.size() && k==line_R.size() )
	{
		for(;i<line_B.size();i=i+2)
		{
			coordinateX=line_B.at(i);
			coordinateY=line_B.at(i+1);
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=255;  //蓝色
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
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //绿色
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
			BRMat.at<Vec3b>(coordinateX,coordinateY)[0]=0;  //红色
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

//最小二乘法，输出拟合直线的k,b（y=kx+b）
std::vector<double> leastSquareFitting(std::vector<Point> &rPoints)
{
	std::vector<double> resLine(2);  //保存k和b
	int num_points=rPoints.size();
	//std::valarray是表示和操作数值数组的类。它支持元素级别的数学运算和各种形式的广义下标运算符
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
	//根据最小二乘法的公式，计算而得。详见4.12纸质笔记
	A=(data_x*data_x).sum();
	B=data_x.sum();
	C=(data_x*data_y).sum();
	D=data_y.sum();

	float k,b,temp=0;
	if(temp=(A*data_x.size()-B*B))  //分母不为0
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

//画出拟合的直线
//第一个参数：得到的BR最小值点；第二个参数：在一条线上的点；第三个参数：在这条线之外的点
int regressionPoint(const std::vector<Point> &rPoints, std::vector<Point> &rInlinePoints, std::vector<Point> &rOutlinePoints)
{
	int length=rPoints.size();
	std::vector<bool> is_inline;
	for(int i=0;i<length;i++)
		is_inline.push_back(true);

	double avg=0.0;
	std::vector<double> kb;
	bool find_new_points=false;
	while(!find_new_points)  //线内点
	{
		std::vector<int> InlinePointsIndex;
		rInlinePoints.clear();
		for(int i=0;i<length;++i)
		{
			if(is_inline[i])  //if(1):执行
			{
				rInlinePoints.push_back(rPoints[i]);
				InlinePointsIndex.push_back(i);
			}
		}

		//拟合线y=kx+b，计算k、b
		kb=leastSquareFitting(rInlinePoints);

		//残差
		std::vector<double> def;
		for(int i=0;i<rInlinePoints.size();++i)
		{
			def.push_back(abs(kb[0]*rInlinePoints[i].x+kb[1]-rInlinePoints[i].y));
			avg+=def[i];
		}

		//判断是否有新的线外的点
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

	//线外点
	rOutlinePoints.clear();
	for(int i=0;i<length;++i)
	{
		if(!is_inline[i])  //if中为0，执行
			rOutlinePoints.push_back(rPoints[i]);
	}

	//-------------------------------画图------------------------------
	//画图
	//cv::Mat cdst=cv::Mat::zeros(WIDTH,HEIGHT,CV_8SC3);
	Mat cdst=BRMat.clone();
	//画图线内点:黄色
	for(int i=0;i<rInlinePoints.size();i++)
	{
		cv::Point center=cv::Point((int)rInlinePoints[i].x, (int)rInlinePoints[i].y);
		//画圆
		//第一个参数：要画的圆所在的图像
		//第二个参数：圆心坐标
		//第三个参数：半径
		//第四个参数：颜色
		//第五个参数：如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充
		circle(cdst,center,1,cv::Scalar(0,255,255),-1);
	}

	//画出线外点：粉色
	for(int i=0;i<rOutlinePoints.size();i++)
	{
		cv::Point center=cv::Point((int)rOutlinePoints[i].x, (int)rOutlinePoints[i].y);
		circle(cdst, center, 3, cv::Scalar(255,0,255), -1);
	}
	//画线:白色
	cv::Point pt1,pt2;
	int beginDot=dotPoint[0].x, endDot=dotPoint[dotPoint.size()-1].x;
	pt1.x=beginDot;
	pt1.y=kb[0]*pt1.x+kb[1];
	pt2.x=endDot;
	pt2.y=kb[0]*pt2.x+kb[1];
	//第一个参数：要画的线所在的图像
	//第二个参数：直线起点
	//第三个参数：直线终点
	//第四个参数：颜色
	//第五个参数：线条粗细
	//第六个参数：line_type=8，（8邻接)连接 线
	line(cdst, pt1, pt2, cv::Scalar(255,255,255), 1, 8);

	imshow("BRline", cdst);
	cv::namedWindow("BRline",WINDOW_NORMAL);
	cv::waitKey(0);
	imwrite("F:\\Code\\BRLine\\Data\\BRLine\\201702281110043_BRLine.bmp", cdst);

	//输出线内点、线外点、线的kb值  
	cout<<"线内点："<<endl;
    for(auto w:rInlinePoints) 
		std::cout<<w.x<<","<<w.y<<std::endl;  
    std::cout<<std::endl;
	cout<<"线外点："<<endl;
    for(auto w:rOutlinePoints) 
		std::cout<<w.x<<","<<w.y<<std::endl;  
    std::cout<<std::endl;  
	cout<<"k和b："<<endl;
    std::cout<<kb[0]<<","<<kb[1]<<std::endl;  

	return 0;
}

int drawLine()
{
	vector<Point> in_line_points, rOutlinePoints;
	regressionPoint(dotPoint, in_line_points, rOutlinePoints);

	return 0;
}
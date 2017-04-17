#include <iostream>
#include <vector>
#include <highgui/highgui.hpp>
using namespace cv;

int shadowExtraction();  //从场景图和背景图中提取阴影，并计算有效阴影区中每个像素的BR
int findDot();  //找到BR最小值点
std::vector<double> leastSquareFitting(std::vector<Point> &rPoints); //最小二乘法，输出拟合直线的k,b（y=kx+b）
int inserSort(std::vector<Point> &vec);   //插入排序：降序
int regressionPoint(const std::vector<Point> &rPoints, std::vector<Point> &rInlinePoints, std::vector<Point> &rOutlinePoints);//画出拟合的直线
int drawLine();  //调用画线函数regressionPoint


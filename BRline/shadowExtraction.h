#include <iostream>
#include <vector>
#include <highgui/highgui.hpp>
using namespace cv;

int shadowExtraction();  //�ӳ���ͼ�ͱ���ͼ����ȡ��Ӱ����������Ч��Ӱ����ÿ�����ص�BR
int findDot();  //�ҵ�BR��Сֵ��
std::vector<double> leastSquareFitting(std::vector<Point> &rPoints); //��С���˷���������ֱ�ߵ�k,b��y=kx+b��
int inserSort(std::vector<Point> &vec);   //�������򣺽���
int regressionPoint(const std::vector<Point> &rPoints, std::vector<Point> &rInlinePoints, std::vector<Point> &rOutlinePoints);//������ϵ�ֱ��
int drawLine();  //���û��ߺ���regressionPoint


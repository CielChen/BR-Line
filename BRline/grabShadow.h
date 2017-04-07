#include <iostream>
#include <highgui/highgui.hpp>

void onMouse(int event, int x, int y, int flags, void* param);  //判断鼠标动作
int cutShadow();  //将已经得到的阴影+物体图进行处理，Grabcut算法只保留有效的阴影
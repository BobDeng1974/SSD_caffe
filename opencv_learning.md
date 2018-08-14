

查看opencv 版本

```
pkg-config --modversion opencv
```

创建1矩阵

```
cv::Mat im_expand = cv::Mat::ones(cv::Size(512,512),CV_8UC3);
```

```c++

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
 
using namespace std;
using namespace cv;
 
 
int main(int argc, const char * argv[]) {
    // insert code here...
    int sz[] = { 3, 4, 6};  //倒着存放，对应三维立体图像的高、宽、长
    Mat Img = Mat(3,sz, CV_16SC4, Scalar::all(0));
    cout<<"三维图像的维度："<<Img.dims<<endl;
    cout<<"三维图像的通道数："<<Img.channels()<<endl;
    
    cout<<"三维图像的长："<<6<<endl;
    cout<<"三维图像的宽："<<Img.size().width<<endl;
    cout<<"三维图像的高："<<Img.size().height<<endl;
    
    cout<<"存放一个数字的大小 = sizeof（short） = 16/8 = "<<Img.elemSize1()<<endl;
    cout<<"存放一个像素点的大小："<<Img.elemSize()<<endl;
    
    cout<<"平面的大小 -> Img.step[0] = "<<Img.step[0]<<endl;  //长 ✖️ 宽 ✖️ elemSize() = 6 * 4 * 8 = 192
    cout<<"行的大小   -> Img.step[1] = "<<Img.step[1]<<endl;  //长 ✖️ elemSize() = 48
    cout<<"元素的大小 -> Img.step[2] = "<<Img.step[2]<<endl;  //元素的大小
    
    /*寻址方法*/
    unsigned char * addrM012 = NULL;
    addrM012 = Img.data + Img.step[0] * 0 + Img.step[1] * 1 + Img.step[2] * 2;
    cout<<"数据开始地址："<<(void*)Img.data<<endl;
    cout<<"Img[0,1,2] 的地址是："<<(void*)addrM012<<endl;
    return 1;

```



```
 // 创建一个 全为0 的 mat
  cv::Mat im_expand = cv::Mat(cv::Size(512,512),CV_8UC3);
  // int sz[]= { 3, 512, 512 }; 
  // Mat im_expand = Mat(3, sz, CV_32F, Scalar::all(0)); 

  // LOG(INFO)<<"三维图像的维度："<<im_expand.size().width<<endl;
  // LOG(INFO)<<"三维图像的维度："<<im_expand.size().height<<endl;
  for(int i=0;i<im_expand.rows;i++)
  {
    if(i >= x_begin && i <= image_rsz.rows){
      for(int j=0;j<im_expand.cols;j++){
        if(j >= y_begin && j <= image_rsz.c){
          // for(int x = 0; x < image_rsz.rows; x++){
          //   for(int y = 0; y < image_rsz.cols; y++){
            im_expand.at<Vec3b>(i,j)[0]=image_rsz.at<Vec3b>(i - x_begin, j - y_begin)[0];
            im_expand.at<Vec3b>(i,j)[1]=image_rsz.at<Vec3b>(i - x_begin, j - y_begin)[1];
            im_expand.at<Vec3b>(i,j)[2]=image_rsz.at<Vec3b>(i - x_begin, j - y_begin)[2]; 
         
      }
    }
  }
}
```

vector<BBoxData<Dtype> >* boxes

vector<BBoxData<Dtype> >& boxes 的区别
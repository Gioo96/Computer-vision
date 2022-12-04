//
//  main.cpp
//  Lab_1
//
//  Created by Gioel Adriano Vencato on 19/03/21.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NEIGHBORHOOD_Y 9
#define NEIGHBORHOOD_X 9
#define MAX_B_CHANNEL 70
#define MAX_G_CHANNEL 100
#define MAX_R_CHANNEL 70
#define thresh = 20;
using namespace cv;
using namespace std;

void onMouse(int event, int x, int y, int f, void* userdata);

int main(int argc, const char * argv[]) {
    Mat input_img = imread("//Users//gioel//Documents//Control System Engineering//Computer vision//Lab_2//che.png");
    //"//Users//gioel//Documents//Control System Engineering//Computer vision//Lab_1//robocup.jpg"
    namedWindow("img");
    imshow("img", input_img);
    setMouseCallback("img", onMouse, static_cast<void*>(&input_img));
    waitKey(0);
    //    Mat input_img = imread("//Users//gioel//Documents//Control System Engineering//Computer vision//Lab_1//robocup.jpg");
//    setMouseCallback("image", onMouse, (void*)&input_img);
//    namedWindow("RobCup");
//    imshow("RobCupp",input_img);
//    waitKey(0);
//
//    return 0;
}
void onMouse(int event, int x, int y, int f, void* userdata) { // x:col y:row
    if (event == EVENT_LBUTTONDOWN) {
      Mat image = *static_cast<Mat*>(userdata);
      Mat image_out = image.clone();
      if (y + NEIGHBORHOOD_Y  > image_out.rows || x + NEIGHBORHOOD_X > image_out.cols)
        return;
      double b = 0;
      double g = 0;
      double r = 0;
      for (int i=y;i < NEIGHBORHOOD_Y + y;i++) {
       for (int j=x;j < NEIGHBORHOOD_X + x;j++) {
        b = b + image.at<Vec3b> (i,j)[0];
        g = g + image.at<Vec3b> (i,j)[1];
        r = r + image.at<Vec3b> (i,j)[2];
       }
      }
        b = b/(NEIGHBORHOOD_Y*NEIGHBORHOOD_X);
        g = g/(NEIGHBORHOOD_Y*NEIGHBORHOOD_X);
        r = r/(NEIGHBORHOOD_Y*NEIGHBORHOOD_X);
    //namedWindow("img_out");
    //imshow("img_out",image_out);
    //waitKey(0);
    vector<double> M(3);
    M.at(0) = b;
    M.at(1) = g;
    M.at(2) = r;
       double mea = (b+g+r)/3;
    for (int i = 0;i < image.rows;i++) {
        for(int j = 0;j < image.cols;j++) {
           if ((image.at<Vec3b>(i,j)[0] < (M.at(0) - mea) || image.at<Vec3b>(i,j)[0] > (M.at(0) + mea)) && (image.at<Vec3b>(i,j)[1] < (M.at(1) - mea) || image.at<Vec3b>(i,j)[0] > (M.at(1) + mea)) && (image.at<Vec3b>(i,j)[2] < (M.at(0) - mea) || image.at<Vec3b>(i,j)[2] > (M.at(0) + mea))) {
               image_out.at<Vec3b>(i,j)[0] = 0;
               image_out.at<Vec3b>(i,j)[1] = 0;
               image_out.at<Vec3b>(i,j)[2] = 0;
           }
        }
    }
        namedWindow("img_out");
        imshow("img_out",image_out);
        waitKey(0);
    
    //Rect rect(x,y,NEIGHBORHOOD_X,NEIGHBORHOOD_Y);
    //Scalar mean = cv::mean(image_out(rect));
    //cout <<"The real mean is"<< mean << endl;
    cout<<"The mean is "<<M.at(0)<<" "<<M.at(1)<<" "<<M.at(2)<<endl;
}
//    for (int i=y;i < NEIGHBORHOOD_Y+y;i++) {
//       for(int j=x;j < NEIGHBORHOOD_X+x;j++) {
//          image_out.at<Vec3b> (i,j)[0] = 255;
//          image_out.at<Vec3b> (i,j)[1] = 255;
//          image_out.at<Vec3b> (i,j)[2] = 255;
//       }
//    }

}

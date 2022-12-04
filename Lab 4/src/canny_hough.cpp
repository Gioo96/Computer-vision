#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "canny_hough.h"

using namespace cv;

// Base Class
// constructor
Base::Base(cv::Mat input_img) {

    input_image = input_img;
}

// doAlgorithm
void Base::doAlgorithm() {
    
    result_image = input_image;
}


// Canny_edge Class
// Constructor
Canny_edge::Canny_edge(cv::Mat input_img, int th1, int th2, int apertureSize) : Base(input_img) {
    
    if (apertureSize % 2 == 0)
        apertureSize++;
    aperture_size_Canny = apertureSize;
    threshold1_Canny = th1;
    threshold2_Canny = th2;
}

// doAlgorithm
void Canny_edge::doAlgorithm() {
    
    cv::Canny(input_image, result_image, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
}

// setThreshold1
void Canny_edge::setThreshold1(int th1) {
    
    threshold1_Canny = th1;
}

// setThreshold2
void Canny_edge::setThreshold2(int th2) {
    
    threshold2_Canny = th2;
}

// getThreshold1
int Canny_edge::getThreshold1() {
    
    return threshold1_Canny;
}

// getThreshold2
int Canny_edge::getThreshold2() {
    
    return threshold2_Canny;
}

cv::Mat Canny_edge::getResult() {
    
    return result_image;
}


// HoughLine Class
// Constructor
HoughLine::HoughLine(cv::Mat input_img, int rho, double theta, int threshold) : Base(input_img) {
    
    rho_HoughLine = rho;
    theta_HoughLine = theta;
    threshold_HoughLine = threshold;
}

// doAlgorithm
void HoughLine::doAlgorithm() {
    
    cv::HoughLines(input_image, lines, rho_HoughLine, theta_HoughLine, threshold_HoughLine);
    Mat cdst;
    cv::cvtColor(input_image, cdst, COLOR_GRAY2BGR);
    for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( cdst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
        }
    result_image = cdst;
}

// Set rho for HoughLine
void HoughLine::setRho(int rho) {
    
    rho_HoughLine = rho;
}

// Set theta for HoughLine
void HoughLine::setTheta(double theta) {
    
    theta_HoughLine = theta;
}

// Set threshold for HoughLine
void HoughLine::setThreshold(int threshold) {
    
    threshold_HoughLine = threshold;
}

// Get rho
int HoughLine::getRho() {
    
    return rho_HoughLine;
}


// Get theta
double HoughLine::getTheta() {
    
    return theta_HoughLine;
}

// Get threshold
int HoughLine::getThreshold() {
    
    return threshold_HoughLine;
}

std::vector<cv::Vec2f> HoughLine::getLines() {
    
    return lines;
}

cv::Mat HoughLine::getResult() {
    
    return result_image;
}


// HoughCircle Class
// Constructor
HoughCircle::HoughCircle(cv::Mat input_img, int dp, int minDist) : Base(input_img) {
    
    dp_HoughCircle = dp;
    minDist_HoughCircle = minDist;
}

// doAlgorithm
void HoughCircle::doAlgorithm() {
    
    cv::HoughCircles(input_image, circles, HOUGH_GRADIENT, dp_HoughCircle, minDist_HoughCircle,800,30,0,20);
    Mat cdst;
    cv::cvtColor(input_image, cdst, COLOR_GRAY2BGR);
    for( size_t i = 0; i < circles.size(); i++ ) {
             Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
             int radius = cvRound(circles[i][2]);
             // draw the circle outline
             circle( cdst, center, radius, Scalar(0,255,0), FILLED, FILLED, 0 );
    }
    result_image = cdst;
}

// Set dp (inverse ratio of the accumulator resolution) for HoughCircle
void HoughCircle::setDp(int dp) {
    
    dp_HoughCircle = dp;
}

// Set minDist (minimum distance between the centre of detected circles) for HoughCircle
void HoughCircle::setMinDist(int minDist) {
    
    minDist_HoughCircle = minDist;
}

// Get dp
int HoughCircle::getDp() {
    
    return dp_HoughCircle;
}

// Get minDist
int HoughCircle::getMinDist() {
    
    return minDist_HoughCircle;
}

// getCircles
std::vector<cv::Vec3f> HoughCircle::getCircles() {
    
    return circles;
}

cv::Mat HoughCircle::getResult() {
    
    return result_image;
}

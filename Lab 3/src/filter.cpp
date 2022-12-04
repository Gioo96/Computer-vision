#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

using namespace cv;

// constructor
Filter::Filter(cv::Mat input_img, int size) {

    input_image = input_img;
    if (size % 2 == 0)
        size++;
    filter_size = size;
}

// for base class do nothing (in derived classes it performs the corresponding filter)
void Filter::doFilter() {

    // it just returns a copy of the input image
    result_image = input_image.clone();

}

// get output of the filter
cv::Mat Filter::getResult() {

    return result_image;
}

//set window size (it needs to be odd)
void Filter::setSize(int size) {

    if (size % 2 == 0)
        size++;
    filter_size = size;
}

//get window size
int Filter::getSize() {

    return filter_size;
}



// Write your code to implement the Gaussian, median and bilateral filters
// Gaussian Filter
// constructor
GaussianFilter::GaussianFilter(cv::Mat input_img, int size, int sigma) : Filter(input_img, size) {
 
    filter_sigma = sigma;
}

void GaussianFilter::doFilter() {

    GaussianBlur(input_image, result_image, Size(filter_size,filter_size), filter_sigma, 0);
}

void GaussianFilter::setSigma(int sigma) {
    
    filter_sigma = sigma;
}

int GaussianFilter::getSigma() {
    
    return filter_sigma;
}

// Median Filter
// constructor
MedianFilter::MedianFilter(cv::Mat input_img, int size) : Filter(input_img, size) {

}

void MedianFilter::doFilter() {
    
    medianBlur(input_image, result_image, filter_size);
}

// Bilateral Filter
// constructor
BilateralFilter::BilateralFilter(cv::Mat input_img, int size, int sigma_range, int sigma_space) : Filter(input_img, size) {

    filter_sigma_range = sigma_range;
    filter_sigma_space = sigma_space;

}

void BilateralFilter::doFilter() {

    bilateralFilter(input_image, result_image, filter_size, filter_sigma_range, filter_sigma_space);
}

void BilateralFilter::setSigma_range(int sigma_range) {
    
    filter_sigma_range = sigma_range;
}

int BilateralFilter::getSigma_range() {
    
    return filter_sigma_range;
}

void BilateralFilter::setSigma_space(int sigma_space) {
    
    filter_sigma_space = sigma_space;
}

int BilateralFilter::getSigma_space() {
    
    return filter_sigma_space;
}

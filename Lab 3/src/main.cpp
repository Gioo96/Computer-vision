#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include "filter.cpp"

using namespace std;
using namespace cv;

// Function headers
void showHistogram_Image(vector<Mat>& hists, Mat src, Mat& dst, String name);
vector<Mat> getEqualized_hist(vector<Mat> bgr_planes, Mat& eq_img);
void showEqualizedHSV(vector<Mat> hsv_planes, String equalization, Mat& bgr);
void onGaussianFilterKernelSize(int pos, void *userdata);
void onGaussianFilterSigma(int pos, void *userdata);
void onMedianFilterKernelSize(int pos, void *userdata);
void onBilateralFilterKernelSize(int pos, void *userdata);
void onBilateralFilterSigmaRange(int pos, void *userdata);
void onBilateralFilterSigmaSpace(int pos, void *userdata);

int main(int argc, const char * argv[]) {

    // Load image
    
    Mat src = imread(argv[1], IMREAD_COLOR);
    //Mat src = imread("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_3/img/image.jpg", IMREAD_COLOR);
    if (!src.data) {
        cout<<"Error loading the image"<<endl;
        return -1;
    }
    Size size(750,400);
    resize(src, src, size);
    namedWindow("Input image");
    imshow("Input image", src);
    waitKey(0);
    
    // Get BGR-channels and corresponding histograms
    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat hist_b, hist_g, hist_r;
    calcHist( &bgr_planes.at(0), 1, 0, Mat(), hist_b, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes.at(1), 1, 0, Mat(), hist_g, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes.at(2), 1, 0, Mat(), hist_r, 1, &histSize, &histRange, uniform, accumulate );

    // Show (histograms of BGR + input image)
    vector<Mat> hists;
    hists.push_back(hist_b);
    hists.push_back(hist_g);
    hists.push_back(hist_r);
    Mat dst1;
    showHistogram_Image(hists, src, dst1, "Histograms, source image");
   
    // Show (histograms of equalized BGR), Show (equalized image)
    Mat eq_img; // Equalized image
    vector<Mat> hists_eq; // Histograms of the equalized image
    hists_eq = getEqualized_hist(bgr_planes, eq_img);
    Mat dst2;
    showHistogram_Image(hists_eq, eq_img, dst2, "Histograms, equalized source image");
    
    // From BGR to HSV color space
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    
    // Get HSV-channels
    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);

    // Equalize only HUE and display the resulting image in the RGB color space
    Mat bgr_h;
    showEqualizedHSV(hsv_planes, "Hue", bgr_h);
    
    // Equalize only SATURATION and display the resulting image in the RGB color space
    Mat bgr_s;
    showEqualizedHSV(hsv_planes, "Saturation", bgr_s);
    
    // Equalize only VALUE and display the resulting image in the RGB color space
    Mat bgr_v;
    showEqualizedHSV(hsv_planes, "Value", bgr_v);
    
    // Show (histograms), Show (image: equalized Value(V) in HSV color space)
    vector<Mat> bgr_planes_v;
    split(bgr_v, bgr_planes_v);
    vector<Mat> hists_eq_val; // Histograms of the equalized image
    Mat eq_img_v;
    hists_eq_val = getEqualized_hist(bgr_planes_v, eq_img_v);
    Mat dst;
    showHistogram_Image(hists_eq_val, bgr_v, dst, "Histograms, equalized V channel");
    
    // Parameters needed for Gaussian, Median, Bilateral filters
    int k_size = 9;
    int sigma = 20;
    int sigma_range = 50;
    int sigma_space = 6;
    
    int count_size = 31;
    int count_sigma = 30;
    int count_sigma_range = 50;
    int count_sigma_space = 50;
    
    // Gaussian Filter
    Mat src_gaussian = bgr_v.clone();
    GaussianFilter gaussian(src_gaussian, k_size, sigma);
    gaussian.doFilter();
    namedWindow("Gaussian filter", WINDOW_AUTOSIZE);
    createTrackbar("Kernel size", "Gaussian filter", &k_size, count_size);
    createTrackbar("Sigma", "Gaussian filter", &sigma, count_sigma);
    imshow("Gaussian filter",gaussian.getResult());
    createTrackbar("Kernel size", "Gaussian filter", &k_size, count_size, onGaussianFilterKernelSize,static_cast<void*>(&gaussian));
    createTrackbar("Sigma", "Gaussian filter", &sigma, count_sigma, onGaussianFilterSigma,static_cast<void*>(&gaussian));
    waitKey(0);
    
    // Median Filter
    Mat src_median = bgr_v.clone();
    MedianFilter median(src_median, k_size);
    median.doFilter();
    namedWindow("Median filter", WINDOW_AUTOSIZE);
    createTrackbar("Kernel size", "Median filter", &k_size, count_size);
    imshow("Median filter",median.getResult());
    createTrackbar("Kernel size", "Median filter", &k_size, count_size, onMedianFilterKernelSize,static_cast<void*>(&median));
    waitKey(0);
    
    // Bilateral Filter
    Mat src_bilateral = bgr_v.clone();
    BilateralFilter bilateral(src_bilateral, k_size, sigma_range, sigma_space);
    bilateral.doFilter();
    imwrite("bilateral2.png", bilateral.getResult());
    namedWindow("Bilateral filter", WINDOW_AUTOSIZE);
    createTrackbar("Sigma space", "Bilateral filter", &sigma_space, count_sigma_space);
    createTrackbar("Sigma range", "Bilateral filter", &sigma_range, count_sigma_range);
    createTrackbar("Kernel size", "Bilateral filter", &k_size, count_size);
    imshow("Bilateral filter",bilateral.getResult());
    createTrackbar("Sigma space", "Bilateral filter", &sigma_space, count_sigma_space, onBilateralFilterSigmaSpace,static_cast<void*>(&bilateral));
    createTrackbar("Sigma range", "Bilateral filter", &sigma_range, count_sigma_range, onBilateralFilterSigmaRange,static_cast<void*>(&bilateral));
    createTrackbar("Kernel size", "Bilateral filter", &k_size, count_size, onBilateralFilterKernelSize,static_cast<void*>(&bilateral));
    waitKey(0);
    
    return 0;
}

// hists: histrograms
// src : source image
// dst: final image (histrograms + source image)
void showHistogram_Image(vector<Mat>& hists, Mat src, Mat& dst, String name)
{
    // Min/Max computation
    double hmax[3] = {0,0,0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = { "blue", "green", "red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                           cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());
    
    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++) {
    
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++) {
     
            cv::line(canvas[i], cv::Point(j, rows), cv::Point(j, rows - (hists[i].at<float>(j) *rows/hmax[i])), hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i], 1, 8, 0);
        }
    }
    // Concatenate the 3 canvas (vertically)
    vconcat(canvas, dst);
    // Resize
    Size size_hist(350,src.rows);
    resize(dst, dst, size_hist);
    // Concatenate the histograms + source image (horizontally)
    hconcat(dst, src, dst);
    // Show the result
    namedWindow(name);
    imshow(name, dst);
    waitKey(0);
  
}

vector<Mat> getEqualized_hist(vector<Mat> bgr_planes, Mat& eq_img) {
    
    // Get equalized BGR-channels
    Mat histim_b, histim_g, histim_r;
    equalizeHist(bgr_planes.at(0), histim_b);
    equalizeHist(bgr_planes.at(1), histim_g);
    equalizeHist(bgr_planes.at(2), histim_r);
    
    // Equalized image
    vector<Mat> histim_eq;
    histim_eq.push_back(histim_b);
    histim_eq.push_back(histim_g);
    histim_eq.push_back(histim_r);
    merge(histim_eq, eq_img);
    
    // Get histograms of equalized BGR
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat histeq_b, histeq_g, histeq_r;
    calcHist( &histim_eq.at(0), 1, 0, Mat(), histeq_b, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &histim_eq.at(1), 1, 0, Mat(), histeq_g, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &histim_eq.at(2), 1, 0, Mat(), histeq_r, 1, &histSize, &histRange, uniform, accumulate );
    vector<Mat> hists_eq;
    hists_eq.push_back(histeq_b);
    hists_eq.push_back(histeq_g);
    hists_eq.push_back(histeq_r);
    
    return hists_eq;
}

void showEqualizedHSV(vector<Mat> hsv_planes, String equalization, Mat& bgr) {
    
    // Equalize hue, merge with saturation and value, show the result
    // Equalize sat, merge with hue and value, show the result
    // Equalize val, merge with hue and saturation, show the result
    
    // Equalize hue, sat, val and put them together into histim_eq_hsv
    Mat histim_h, histim_s, histim_v;
    equalizeHist(hsv_planes.at(0), histim_h);
    equalizeHist(hsv_planes.at(1), histim_s);
    equalizeHist(hsv_planes.at(2), histim_v);
    
    if (equalization == "Hue") {
        
        // Merge equalized hue with sat and val
        vector<Mat> histim_eq_h;
        Mat h_eq;
        histim_eq_h.push_back(histim_h);
        histim_eq_h.push_back(hsv_planes.at(1));
        histim_eq_h.push_back(hsv_planes.at(2));
        merge(histim_eq_h, h_eq);
    
        // Go back to BGR color space
        cvtColor(h_eq, bgr, COLOR_HSV2BGR);
        namedWindow("Equalized Hue");
        imshow("Equalized Hue", bgr);
        waitKey(0);
    }
    
    else if (equalization == "Saturation") {
        
        // Merge equalized sat with hue and val
        vector<Mat> histim_eq_s;
        Mat s_eq;
        histim_eq_s.push_back(hsv_planes.at(0));
        histim_eq_s.push_back(histim_s);
        histim_eq_s.push_back(hsv_planes.at(2));
        merge(histim_eq_s, s_eq);
        
        // Go back to BGR color space
        cvtColor(s_eq, bgr, COLOR_HSV2BGR);
        namedWindow("Equalized Sat");
        imshow("Equalized Sat", bgr);
        waitKey(0);
    }
    
    else if (equalization == "Value") {
        
        // Merge equalized val with hue and sat
        vector<Mat> histim_eq_v;
        Mat v_eq;
        histim_eq_v.push_back(hsv_planes.at(0));
        histim_eq_v.push_back(hsv_planes.at(1));
        histim_eq_v.push_back(histim_v);
        merge(histim_eq_v, v_eq);
        
        // Go back to BGR color space
        cvtColor(v_eq, bgr, COLOR_HSV2BGR);
        namedWindow("Equalized Val");
        imshow("Equalized Val", bgr);
        waitKey(0);
    }
    
    else {
        
        cout<<"Insert valid channel name!"<<endl;
        return;
    }
}

void onGaussianFilterKernelSize(int pos, void *userdata) {

    GaussianFilter* gaussian = static_cast<GaussianFilter*>(userdata);
    gaussian->setSize(pos);
    cout<<"Kernel size: " + to_string(gaussian->getSize())<<endl;
    cout<<"Sigma: " + to_string(gaussian->getSigma())<<endl;
    gaussian->doFilter();
    imshow("Gaussian filter", gaussian->getResult());
}

void onGaussianFilterSigma(int pos, void *userdata) {

    GaussianFilter* gaussian = static_cast<GaussianFilter*>(userdata);
    gaussian->setSigma(pos);
    cout<<"Kernel size: " + to_string(gaussian->getSize())<<endl;
    cout<<"Sigma: " + to_string(gaussian->getSigma())<<endl;
    gaussian->doFilter();
    imshow("Gaussian filter", gaussian->getResult());
}

void onMedianFilterKernelSize(int pos, void *userdata) {

    MedianFilter* median = static_cast<MedianFilter*>(userdata);
    median->setSize(pos);
    cout<<"Kernel size: " + to_string(median->getSize())<<endl;
    median->doFilter();
    imshow("Median filter", median->getResult());
}

void onBilateralFilterKernelSize(int pos, void *userdata) {

    BilateralFilter* bilateral = static_cast<BilateralFilter*>(userdata);
    bilateral->setSize(pos);
    cout<<"Kernel size: " + to_string(bilateral->getSize())<<endl;
    cout<<"Sigma range: " + to_string(bilateral->getSigma_range())<<endl;
    cout<<"Sigma space: " + to_string(bilateral->getSigma_space())<<endl;
    bilateral->doFilter();
    imshow("Bilateral filter", bilateral->getResult());
}

void onBilateralFilterSigmaRange(int pos, void *userdata) {

    BilateralFilter* bilateral = static_cast<BilateralFilter*>(userdata);
    bilateral->setSigma_range(pos);
    cout<<"Kernel size: " + to_string(bilateral->getSize())<<endl;
    cout<<"Sigma range: " + to_string(bilateral->getSigma_range())<<endl;
    cout<<"Sigma space: " + to_string(bilateral->getSigma_space())<<endl;
    bilateral->doFilter();
    imshow("Bilateral filter", bilateral->getResult());
}

void onBilateralFilterSigmaSpace(int pos, void *userdata) {

    BilateralFilter* bilateral = static_cast<BilateralFilter*>(userdata);
    bilateral->setSigma_space(pos);
    cout<<"Kernel size: " + to_string(bilateral->getSize())<<endl;
    cout<<"Sigma range: " + to_string(bilateral->getSigma_range())<<endl;
    cout<<"Sigma space: " + to_string(bilateral->getSigma_space())<<endl;
    bilateral->doFilter();
    imshow("Bilateral filter", bilateral->getResult());
}

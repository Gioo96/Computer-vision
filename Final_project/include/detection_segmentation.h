#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/photo.hpp>
#include <filesystem>

// Global variables
int click_pair = 0;
std::vector<std::pair<cv::Point, cv::Point>> vec_pairs;
std::pair<cv::Point, cv::Point> p1_p2;
cv::Point p1, p2;

using namespace cv;
using namespace std;

// DATASET CLASS
class Dataset {

public:

    // CONSTRUCTOR
    Dataset(String images_path, String model_CNN_path);
};

// BASE CLASS
class Base {
    
public:

    // CONSTRUCTOR
    Base(String pattern, int fl);
    
// DATA
public:

    // List of test images
    vector<Mat> test_images;
    // Kaggle dataset: flag = 1, Venice dataset: flag = 0
    int flag;
};

// DETECTION CLASS
class Detection : public Base {

public:

    // CONSTRUCTOR
    Detection(String pattern, int fl);
    
    // METHODS
    // detection
    void detection(vector<Mat> image, String model_CNN_pb);
    // rect_return
    vector<Rect> rect_return(vector<Rect> all_rects, vector<Rect> seed_rects, int scaling);
    // getMetrics
    void getMetrics(vector<Rect> predicted_rects, Mat image_predicted, int index);
};

// SEGMENTATION CLASS
class Segmentation : public Base {

public:

    // CONSTRUCTOR
    Segmentation(String pattern, int fl);
    
    // METHODS
    // segmentation
    void segmentation(String ground_truth_segmentation_path);
    // pixel_accuracy metric
    double getMetrics(String ground_truth_images, Mat segmentated_images, int index);
    //swap_colors
    void swap_colors(Mat& image);
    //click
    void click(Mat image);
    //onMouse
    static void onMouse(int event, int x, int y, int f, void* userdata);
};

// HOUGH LINE CLASS
class HoughLine{

public:

    // CONSTRUCTOR
    HoughLine(cv::Mat input_image, int rho, double theta, int threshold);
    
    // METHODS
    // Performs Hough line detector algorithm
    void doAlgorithm();
    // Set rho for HoughLine
    void setRho(int rho);
    // Set theta for HoughLine
    void setTheta(double theta);
    // Set threshold for HoughLine
    void setThreshold(int threshold);
    // Get rho
    int getRho();
    // Get theta
    double getTheta();
    // Get threshold
    int getThreshold();
    // getLines
    std::vector<pair<Point, Point>> getLines();
    // getResult
    cv::Mat getResult();
    //onHoughLineThetaden
    static void onHoughLineThetaden(int pos, void *userdata);
    // drawStraightLine
    void drawStraightLine(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Scalar color);
        

// DATA
protected:

    // rho
    int rho_HoughLine;
    // theta
    double theta_HoughLine;
    // threshold
    int threshold_HoughLine;
    std::vector<pair<Point, Point>> lines;
    // Input image
    Mat input_image_hough;
    // Result image
    Mat result_image_hough;
};

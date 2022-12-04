#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Base class
class Base{

// Methods

public:

    // Constructor
    Base(cv::Mat input_img);
    
    // Algorithm
    void doAlgorithm();

// Data

protected:

    // Input image
    cv::Mat input_image;

    // Output image
    cv::Mat result_image;

};

// Class implementing the Canny edge detector
class Canny_edge : public Base {

// Methods

public:

    // Constructor
    Canny_edge(cv::Mat input_image, int th1, int th2, int apertureSize);
    
    // Performs Canny edge detector algorithm
    void doAlgorithm();
    
    // Set threshold1 for Canny
    void setThreshold1(int th1);
    
    // Set threshold2 for Canny
    void setThreshold2(int th2);
    
    // Get threshold1
    int getThreshold1();
    
    // Get threshold2
    int getThreshold2();
    
    // Get result
    cv::Mat getResult();

// Data

protected:

    // Aperture size of the Sobel operator (Canny)
    int aperture_size_Canny;
    
    // threshold_1
    int threshold1_Canny;
    
    // threshold_2
    int threshold2_Canny;
};

// Class implementing the Hough line detector
class HoughLine : public Base {

// Methods

public:

    // Constructor
    HoughLine(cv::Mat input_image, int rho, double theta, int threshold);
    
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
    
    std::vector<cv::Vec2f> getLines();
    
    cv::Mat getResult();
        

// Data

protected:

    // rho
    int rho_HoughLine;
    
    // theta
    double theta_HoughLine;
    
    // threshold
    int threshold_HoughLine;
    
    std::vector<cv::Vec2f> lines;
};

// Class implementing the Hough circle detector
class HoughCircle : public Base {

// Methods

public:

    // Constructor
    HoughCircle(cv::Mat input_image, int dp, int minDist);
    
    // Performs Hough circle detector algorithm
    void doAlgorithm();
    
    // Set dp (inverse ratio of the accumulator resolution) for HoughCircle
    void setDp(int dp);
    
    // Set minDist (minimum distance between the centre of detected circles) for HoughCircle
    void setMinDist(int minDist);
    
    // Get dp
    int getDp();
    
    // Get minDist
    int getMinDist();
    
    // getCircles
    std::vector<cv::Vec3f> getCircles();
    
    cv::Mat getResult();

// Data

protected:

    // dp
    int dp_HoughCircle;
    
    // minDist
    int minDist_HoughCircle;
    
    // circles
    std::vector<cv::Vec3f> circles;
};


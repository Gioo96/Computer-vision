#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// PanoramicImage Class
class Tracking{

// Methods

public:

    // Constructor
    Tracking(vector<Mat> images_f, String dataset_path);
    
    // Visualize good keypoints
    vector<vector<Point2f>> visualizeGoodKeypoints(vector<Mat> &H, Mat &img_keypoints);
    
    // Draw the rectangle
    void drawRect(vector<Mat> H, Mat &img, vector<vector<Point2f>> &corners, int flag);
    
    //Track the objects
    void trackObjects(vector<vector<Point2f>> allActual_corners, vector<vector<Point2f>> allActual_keypoints);

// Data

public:

    // Vector of images of the given video
    vector<Mat> images_frame;
    
    // Vector of images of given dataset
    vector<Mat> images_dataset;
};

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include "panoramicImage.cpp"


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    // Load images
    String dataset = "dolomites";
    PanoramicImage panoramic(dataset);
    
    // Projected images
    vector<Mat> list;
    list = panoramic.cylindricalProj_images(panoramic.getImages());
    
    // Compute Keypoints and Descriptors foreach projected image
    vector<vector<KeyPoint>> list_keypoints;
    vector<Mat> list_descriptors;
    panoramic.Keypoints_Descriptors(list, list_keypoints, list_descriptors);
    
    // Compute all good matches between consecutive images
    vector<vector<DMatch>> good_matches;
    good_matches = panoramic.getMatches(list_descriptors,3);
    
    //
    //
    //
    Mat img;
    drawMatches(list.at(0), list_keypoints.at(0), list.at(1), list_keypoints.at(1), good_matches.at(0), img);
    imshow("aaaa", img);
    waitKey(0);
    
    // Compute the translations between consecutive images
    vector<vector<int>> translations_x_y;
    translations_x_y = panoramic.findTranslations(good_matches, list_keypoints);
    
    // Get panoramic
    Mat result;
    result = panoramic.getPanoramic(list, translations_x_y);
    namedWindow("Panoramic view");
    imshow("Panoramic view", result);
    waitKey(0);
    
    
    return 0;
}

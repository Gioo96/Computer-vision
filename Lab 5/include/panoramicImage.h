#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramic_utils.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
class PanoramicImage{

// Methods

public:

    // Constructor
    PanoramicImage(String data);
    
    // Project the images on a cilinder surface
    vector<Mat> cylindricalProj_images(vector<Mat> images);
    
    // Get the features and descriptors of the projected images
    void Keypoints_Descriptors(vector<Mat> list_projImages, vector<vector<KeyPoint>> &list_keypoints, vector<Mat> &list_descriptors);
    
    // Get the matches between all consecutive projected images
    vector<vector<DMatch>> getMatches(vector<Mat> list_descriptors, double ratio);
    
    // Find translations (in pixels)
    vector<vector<int>> findTranslations(vector<vector<DMatch>> matches, vector<vector<KeyPoint>> keypoints);
    
    // Get panoramic image
    Mat getPanoramic(vector<Mat> images, vector<vector<int>> translations);
    
    // Return the list of images contained in the dataset
    vector<Mat> getImages();

// Data

protected:

    // Vector of images
    vector<Mat> list_images;
    
    // Vector of FOV/2 angles
    // Dolomites dataset: 27°
    // All the other datasets: 33°
    vector<double> angles;
    
    // KeyPoint of all refined matches
    vector<KeyPoint> matches_kp;
    
    // Output image
    Mat result_image;

};

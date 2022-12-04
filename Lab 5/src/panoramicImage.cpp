#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramicImage.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
// constructor
PanoramicImage::PanoramicImage(String data) {

    vector<String> fn;
    if (data == "lab" or data == "kitchen") {
        
        glob("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_5/data/" + data + "/*.bmp", fn, false);
    }
       
    else {
            
        glob("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_5/data/" + data + "/*.png", fn, false);
    }
    
    if (!fn.size()) {
        
        cout<<"Error loading the dataset"<<endl;
        return;
    }
    else {
        
        size_t count = fn.size(); // Number of images
        Size size(1500,1000);
    
        for (size_t i = 0; i < count; i++) {
            
            list_images.push_back(imread(fn[i], IMREAD_COLOR));
            resize(list_images.at(i),list_images.at(i) ,size);
            namedWindow("Image number: " + to_string(i+1));
            imshow("Image number: " + to_string(i+1), list_images.at(i));
            waitKey(1);
            if (data == "dolomites") {
            
                angles.push_back(27);
            }
            else {
            
                angles.push_back(33);
            }
        }
    }

}

vector<Mat> PanoramicImage::cylindricalProj_images(vector<Mat> images) {
    
    vector<Mat> list_projImages;
    for (size_t i = 0; i < images.size(); i ++) {
        Mat img;
        img = PanoramicUtils::cylindricalProj(images.at(i), angles.at(i));
        equalizeHist(img, img);
        list_projImages.push_back(img);
    }
    return list_projImages;
}

void PanoramicImage::Keypoints_Descriptors(vector<Mat> list_projImages, vector<vector<KeyPoint>> &list_keypoints, vector<Mat> &list_descriptors) {
    
    Ptr<SIFT> sift = SIFT::create();
    for (size_t i = 0; i < list_projImages.size(); i ++) {
        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift->detectAndCompute(list_projImages.at(i), Mat(), keypoints, descriptors);
        list_keypoints.push_back(keypoints);
        list_descriptors.push_back(descriptors);
    }
}

vector<vector<DMatch>> PanoramicImage::getMatches(vector<Mat> list_descriptors, double ratio) {
    
    vector<vector<DMatch>> good_matches_allpairs;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    for (int i = 0; i < list_descriptors.size()-1; i ++) {
        
        // Get all matches of 2 consecutive projected images
        vector<DMatch> matches;
        matcher->match(list_descriptors.at(i), list_descriptors.at(i+1), matches);
    
        // Get minimum distance between descriptors
        double min_distance = 200;
        for (int j = 0; j < matches.size(); j++) {

            if (matches.at(j).distance < min_distance && matches.at(j).distance > 0) {

                min_distance = matches.at(j).distance;
            }
        }
    
        // Refine matches
        vector<DMatch> good_matches_between_pairs;
        for (int j = 0; j < matches.size(); j++) {

            if (matches.at(j).distance < ratio*min_distance) {

                good_matches_between_pairs.push_back(matches.at(j));
            }
        }
        good_matches_allpairs.push_back(good_matches_between_pairs);
    }
    
    return good_matches_allpairs;
}

vector<vector<int>> PanoramicImage::findTranslations(vector<vector<DMatch>> matches, vector<vector<KeyPoint>> keypoints) {

    // Find valid matches (RANSAC)
    vector<vector<int>> mask_all;
    vector<vector<Point2f>> src_vec, dst_vec;
    for (int i = 0; i < matches.size(); i++) {
        vector<Point2f> src,dst;
        vector<int> mask; // mask will contain 0 if the match is wrong
        for (int j = 0; j < matches.at(i).size(); j++) {
            
            // Good matches in the first image (2 consecutive images are considered)
            src.push_back(keypoints.at(i).at(matches.at(i).at(j).queryIdx).pt);
            // Good matches in the second image (2 consecutive images are considered)
            dst.push_back(keypoints.at(i+1).at(matches.at(i).at(j).trainIdx).pt);
        }
        src_vec.push_back(src);
        dst_vec.push_back(dst);
        findHomography(src, dst, mask, RANSAC);
        mask_all.push_back(mask);
    }
    
    // Find the translations
    vector<int> avg_x_vec, avg_y_vec;
    for (int i = 0; i < mask_all.size(); i++) {

        double avg_x = 0, avg_y = 0;
        int count = 0;
        for (int j = 0; j < mask_all.at(i).size(); j++) {

            if (mask_all.at(i).at(j) != 0) {

                avg_x += src_vec.at(i).at(j).x - dst_vec.at(i).at(j).x;
                avg_y += src_vec.at(i).at(j).y - dst_vec.at(i).at(j).y;
                count ++;
            }
        }
        avg_x = avg_x/count;
        avg_y = avg_y/count;
        avg_x_vec.push_back(static_cast<int>(avg_x));
        avg_y_vec.push_back(static_cast<int>(avg_y));
    }
    vector<vector<int>> translations_x_y;
    translations_x_y.push_back(avg_x_vec);
    translations_x_y.push_back(avg_y_vec);
    
    return translations_x_y;
}

Mat PanoramicImage::getPanoramic(vector<Mat> images, vector<vector<int>> translations) {
    
    // Compute width of the panoramic image
    int width = 0;
    for (int i = 0; i < images.size(); i++) {

        if (i == images.size()-1) {

            width += images.at(i).cols;
        }
        else {

            width += translations.at(0).at(i);
        }
    }
    
    // Empty panoramic image
    Mat result(images.at(0).rows, width, CV_8UC1, Scalar(0,0,0));
    
    int col_actual = 0, row_actual = 0;
    for (int i = 0; i < images.size(); i++) {

        if (i != images.size() - 1) {
            
            Mat submat = result.operator()(Range::all(), Range(col_actual, col_actual + translations.at(0).at(i)));
            images.at(i).operator()(Range::all(), Range(0, translations.at(0).at(i))).copyTo(submat);
            col_actual += translations.at(0).at(i);
        }
        else {

            Mat submat = result.operator()(Range::all(), Range(col_actual, col_actual + images.at(i).cols));
            images.at(i).operator()(Range::all(), Range(0, images.at(i).cols)).copyTo(submat);
        }
    }
    

    return result;
}

vector<Mat> PanoramicImage::getImages() {
    
    return list_images;
}

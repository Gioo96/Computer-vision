#include "tracking.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
// constructor
Tracking::Tracking(vector<Mat> images_f, String dataset_path) {

    images_frame = images_f;
    dataset_path = dataset_path + "/*.png";
    vector<String> fn;
    glob(dataset_path, fn, false);
    if (!fn.size()) {
        
        cout<<"Error loading the dataset"<<endl;
    }
    else {
        
        size_t count = fn.size(); // Number of images
        for (size_t i = 0; i < count; i++) {
            
            images_dataset.push_back(imread(fn[i], IMREAD_COLOR));
        }
    }
}

vector<vector<Point2f>> Tracking::visualizeGoodKeypoints(vector<Mat> &H, Mat &img_keypoints) {
    
    // COMPUTE KEYPOINTS & DESCRIPTORS
    Ptr<SIFT> sift = SIFT::create();
    vector<Mat> gray_images_dataset(images_dataset.size());
    vector<Mat> detected_edges(images_dataset.size());
    vector<Mat> dst(images_dataset.size());
    int lowThreshold = 0;
    const int ratio_canny = 3;
    const int kernel_size = 3;
    vector<vector<Point2f>> allcoords_keypoints;
    
    // Keypoints & descriptors foreach image of the dataset
    vector<vector<KeyPoint>> list_keypoints_dataset;
    vector<Mat> list_descriptors_dataset;
    for (int i = 0; i < images_dataset.size(); i ++) {
        
        vector<KeyPoint> keypoints_d;
        Mat descriptors_d;
        sift->detectAndCompute(images_dataset.at(i), Mat(), keypoints_d, descriptors_d);
        list_keypoints_dataset.push_back(keypoints_d);
        list_descriptors_dataset.push_back(descriptors_d);
        
        // Try to find the four corners for each image of dataset
        cvtColor(images_dataset.at(i), gray_images_dataset.at(i), COLOR_BGR2GRAY);
    }
    
    // Keypoints & descriptors of the first video frame
    vector<KeyPoint> keypoints_frame;
    Mat descriptors_frame;
    sift->detectAndCompute(images_frame.at(0), Mat(), keypoints_frame, descriptors_frame);

    // GET GOOD KEYPOINTS
    vector<vector<KeyPoint>> allgood_keypoints;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    for (int i = 0; i < images_dataset.size(); i ++) {
        
        // Get all matches of first video frame and image 'i' of the dataset
        vector<DMatch> matches;
        matcher->match(descriptors_frame, list_descriptors_dataset.at(i), matches);
        
        // Get minimum distance between descriptors
        double min_distance = 200;
        for (int j = 0; j < matches.size(); j++) {

            //if (matches.at(j).distance < min_distance && matches.at(j).distance > 0) {
            if (matches.at(j).distance < min_distance) {

                min_distance = matches.at(j).distance;
            }
        }
        
        // Refine matches
        vector<DMatch> refined_matches;
        for (int j = 0; j < matches.size(); j++) {

            if (matches.at(j).distance < 3 * min_distance) {

                refined_matches.push_back(matches.at(j));
            }
        }
        
        // Good matches (RANSAC)
        vector<Point2f> scene, object;
        for (int j = 0; j < refined_matches.size(); j++) {
            
            // Refined matches between the dataset image and first framed img (Pixel cords)
            scene.push_back(keypoints_frame.at(refined_matches.at(j).queryIdx).pt);
            object.push_back(list_keypoints_dataset.at(i).at(refined_matches.at(j).trainIdx).pt);
        }
        
        Mat H_single;
        vector<int> mask; // mask will contain 0 if the match is wrong
        H_single = findHomography(object, scene, mask, RANSAC);
        H.push_back(H_single);

        // Good matches
        vector<DMatch> good_matches;
        for (int j=0; j<mask.size(); j++) {
    
            if (mask.at(j) != 0) {
    
                good_matches.push_back(refined_matches.at(j));
            }
        }
        
        // Good keypoints
        vector<KeyPoint> good_keypoints;
        vector<Point2f> coords_keypoints;
        
        
        for (int j=0; j<good_matches.size(); j++) {
    
            good_keypoints.push_back(keypoints_frame.at(good_matches.at(j).queryIdx));
            coords_keypoints.push_back(good_keypoints.at(j).pt);
            
            //cout <<" Punto"+to_string(j)<< good_keypoints.at(j).pt << endl;
        }
        
        allcoords_keypoints.push_back(coords_keypoints);
        allgood_keypoints.push_back(good_keypoints);
    }
    
    // Visualize good keypoints
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    
    img_keypoints = images_frame.at(0).clone();

    for (int i=0; i<images_dataset.size(); i++) {
        
        drawKeypoints(img_keypoints, allgood_keypoints.at(i), img_keypoints, color.at(i));
    }
    
    imshow("Visualize good keypoints", img_keypoints);
    waitKey(0);
    
    return allcoords_keypoints;
}

void Tracking::drawRect(vector<Mat> H, Mat &img, vector<vector<Point2f>> &corners, int flag) {
         
    // Colors of the rectangles [4 objects max]
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};

    for (int i = 0; i < H.size(); i++) {
    
        // Compute corrispondent pixel in frame image
        perspectiveTransform(corners.at(i), corners.at(i), H.at(i));
    
        // Draw lines
        line(img, corners.at(i)[0], corners.at(i)[1], color.at(i), 3);
        line(img, corners.at(i)[1], corners.at(i)[2], color.at(i), 3);
        line(img, corners.at(i)[2], corners.at(i)[3], color.at(i), 3);
        line(img, corners.at(i)[3], corners.at(i)[0], color.at(i), 3);
        if (flag == 0) {
            
            cout<<"---------"<<endl<<"Press a key"<<endl;
            imshow("Object detection n. " + to_string(i+1), img);
            waitKey(0);
        }
    }
}

void Tracking::trackObjects(vector<vector<Point2f>> allActual_corners, vector<vector<Point2f>> allActual_keypoints) {
    
    // Number of frames
    int size_video = images_frame.size();
    
    // Colors
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    
    Mat first_image_gray; // First frame (gray)
    Mat next_gray, next_image; // Second frame (gray, color)

    // First frame --> gray frame
    cvtColor(images_frame.at(0), first_image_gray, COLOR_BGR2GRAY);
    
    // Termination criteria
    TermCriteria termcrit(TermCriteria::MAX_ITER|TermCriteria::EPS,20,0.03);

    Size subPixWinSize(10,10), winSize(10,10);
    int win_size = 10;
    
    for (int i = 1; i < images_frame.size(); i++) {
        
        vector<uchar> status;
        vector<float> err;
        
        // Copy each frame on next_image and save on next_gray the single channel image
        images_frame.at(i).copyTo(next_image);
        cvtColor(next_image, next_gray, COLOR_BGR2GRAY);
     
        vector<vector<Point2f>> allNext_keypoints; // Pixels coordinates of all keypoints of second frame
        
        // Homography matrix
        vector<Mat> H;
        
        for (int j = 0; j < images_dataset.size(); j++) {
            
            vector<Point2f> nextKeypoints;
            
            cornerSubPix(first_image_gray, allActual_keypoints.at(j), Size(win_size, win_size), Size(-1,-1), termcrit);
            calcOpticalFlowPyrLK(first_image_gray, next_gray, allActual_keypoints.at(j), nextKeypoints, status, err, Size( win_size+1, win_size+1 ), 2, termcrit);
            
            allNext_keypoints.push_back(nextKeypoints);
            
            // Get homography
            Mat H_single;
            H_single = findHomography(allActual_keypoints.at(j), nextKeypoints);
            H.push_back(H_single);
            
            // Circle keypoints
            for (int k = 0; k < nextKeypoints.size(); k++) {
                
                circle(next_image, nextKeypoints.at(k), 3, color.at(j));
            }
        }
        
        // Draw updated rectangles
        drawRect(H, next_image, allActual_corners,1);
        copyTo(next_gray, first_image_gray, Mat());
        //first_image_gray = next_gray;
        allActual_keypoints = allNext_keypoints;
        
        imshow("Video", next_image);
        waitKey(1);
    }
    
}

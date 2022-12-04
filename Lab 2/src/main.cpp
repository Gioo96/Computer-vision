#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#define rows_int 5
#define cols_int 6
#define SQUARE_DIM 0.11

using namespace std;
using namespace cv;

// Function headers
void  Points3D_Points2D(vector<vector<Point3f>>& points3d, vector<vector<Point2f>>& points2d, vector<bool>& patternFound, vector<Mat> images);
vector<double>  reproj_error(vector<vector<Point2f>>& cornerProj, vector<vector<Point3f>> points3d, vector<vector<Point2f>> points2d, vector<Mat> R, vector<Mat> T, Mat cameraMatrix, Mat distCoeffs, vector<Mat> images, int& max_im, int& min_im);


int main(int argc, const char * argv[]) {
    vector<String> fn;
    glob("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_2/img/checkerboard_images/*.png", fn, false);
//    glob(argv[1], fn, false);

    vector<Mat> images;
    size_t count = fn.size(); //number of JPG files in Checkerboard folder
    
    for (size_t i=0; i<count; i++) {
        images.push_back(imread(fn[i], cv::IMREAD_GRAYSCALE));
        namedWindow("Chessboard view");
        imshow("Chessboard view", images.at(i));
        waitKey(1);
    }
    vector<vector<Point3f>> points3d; // Vector of 3D Points foreach image
    vector<vector<Point2f>> points2d; // vector of 2D Points foreach image
    vector<bool> patternFound;

    // 2D Points & 3D Points foreach image
    Points3D_Points2D(points3d,points2d,patternFound,images);
    
    // Check if corners of the first image were correctely found
    Size patternSize(cols_int,rows_int);
    Mat check = images.at(0).clone();
    drawChessboardCorners(check, patternSize, points2d.at(0), patternFound.at(0));
    namedWindow("Check corners");
    imshow("Check corners",check);
    waitKey(0);
 
    // Camera matrix, distorsion coefficients
    Mat cameraMatrix,distCoeffs;
    vector<Mat> R,T;
    vector<double> perViewErrors;
    double rms = calibrateCamera(points3d, points2d, images.at(0).size(), cameraMatrix, distCoeffs, R,T);
    cout<<"The camera matrix is: "<<cameraMatrix<<endl;
    cout<<"The distorsion coefficients are: "<<distCoeffs<<endl;
    
    // Reprojection error foreach image
    vector<vector<Point2f>> cornerProj;
    int max_im = 0;
    int min_im = 0;
    vector<double> vec_reproj_error = reproj_error(cornerProj, points3d, points2d, R, T, cameraMatrix, distCoeffs, images, max_im, min_im);
    double rms_manually = 0;
    double reproj_error_image = 0;
    double reproj_error = 0;
    for (int i=0;i<vec_reproj_error.size();i++) {
        reproj_error_image = sqrt(vec_reproj_error.at(i)*cornerProj.at(0).size());
        rms_manually += vec_reproj_error.at(i);
        reproj_error += reproj_error_image;
    }
    reproj_error = reproj_error/(vec_reproj_error.size());
    rms_manually = sqrt(rms_manually/(vec_reproj_error.size()));
    cout<<"The mean reprojection error is: "<<reproj_error<<endl;
    cout<<"The RMS: "<<rms<<endl;
    cout<<"The RMS computed manually is: "<<rms_manually<<endl;
    
    // Worst image (highest reprejection error)
    namedWindow("Highest reprojection error");
    imshow("Highest reprojection error", images.at(max_im));
    waitKey(0);
    Mat max = images.at(max_im).clone();
    drawChessboardCorners(max, patternSize, cornerProj.at(max_im), patternFound.at(max_im));
    namedWindow("Highest reprojection error_corners");
    imshow("Highest reprojection error_corners",max);
    waitKey(0);
    drawChessboardCorners(max, patternSize, points2d.at(max_im), patternFound.at(max_im));
    namedWindow("Max_err: real corner position");
    imshow("Max_err: real corner position",max);
    waitKey(0);
        
    // Best image (lowest reprojection errror)
    namedWindow("Lowest reprojection error");
    imshow("Lowest reprojection error", images.at(min_im));
    waitKey(0);
    Mat min = images.at(min_im).clone();
    drawChessboardCorners(min, patternSize, cornerProj.at(min_im), patternFound.at(min_im));
    namedWindow("Lowest reprojection error_corners");
    imshow("Lowest reprojection error_corners",min);
    waitKey(0);
    drawChessboardCorners(min, patternSize, points2d.at(min_im), patternFound.at(min_im));
    namedWindow("Min_err: real corner position");
    imshow("Min_err: real corner position",min);
    waitKey(0);
    
    // Undistorted and rectified image
//    Mat dist_img = imread("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_2/img/data/test_image.png");
    Mat dist_img = imread(argv[1]);
    Mat undist_img = dist_img.clone();
    Mat new_cameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, dist_img.size(), 0);
    Mat map1;
    Mat map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), new_cameraMatrix, dist_img.size(),CV_32FC2,map1,map2);
    Mat dstmap1;
    Mat dstmap2;
    remap(dist_img, undist_img, map1, map2, INTER_NEAREST);
    imshow("Distorted image", dist_img);
    imshow("Undistorted image", undist_img);
    waitKey(0);
    
    return 0;
}



void  Points3D_Points2D(vector<vector<Point3f>>& points3d, vector<vector<Point2f>>& points2d, vector<bool>& patternFound, vector<Mat> images) {
    
    // 3D Points (1 image)
    vector<Point3f> points3d_img;
    for (int i=0; i<rows_int; i++) {
        for (int j=0; j<cols_int; j++) {
            points3d_img.push_back(Point3f(i*SQUARE_DIM,j*SQUARE_DIM,0));
        }
    }
    
    TermCriteria termCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    Size patternSize(cols_int,rows_int);
    
    // 2D Points & 3D Points foreach image
    for (int i=0; i<images.size(); i++) {
        vector<Point2f> points2d_img;
        patternFound.push_back(findChessboardCorners(images.at(i), patternSize, points2d_img));
        cornerSubPix(images.at(i), points2d_img, Size(11, 11), Size(-1, -1), termCriteria);
        if (patternFound.at(i)) {
            points2d.push_back(points2d_img);
            points3d.push_back(points3d_img);
        }
    }
    
}

vector<double>  reproj_error(vector<vector<Point2f>>& cornerProj, vector<vector<Point3f>> points3d, vector<vector<Point2f>> points2d, vector<Mat> R, vector<Mat> T, Mat cameraMatrix, Mat distCoeffs, vector<Mat> images, int& max_im, int& min_im) {
    
    for (int i=0;i<images.size();i++) {
        vector<Point2f> cornerProj_im;
        projectPoints(points3d.at(i), R.at(i), T.at(i), cameraMatrix, distCoeffs, cornerProj_im);
        cornerProj.push_back(cornerProj_im);
    }
    
    // Reprojection error
    double max_error = 0;
    double min_error = 0;
    vector<double> vec_e;
    
    for (int im=0;im<images.size();im++) { // Images
        double proj_error = 0;
        proj_error = pow(norm(points2d.at(im),cornerProj.at(im),NORM_L2),2);
        if (im == 0) {
            max_error = proj_error;
            max_im = im;
            min_error = proj_error;
            min_im = im;
        }
        else {
            if (proj_error > max_error) {
                max_error = proj_error;
                max_im = im;
            }
            else if (proj_error < min_error) {
                min_error = proj_error;
                min_im = im;
            }
        }
        vec_e.push_back(proj_error/cornerProj.at(0).size());
    }
   return vec_e;
}

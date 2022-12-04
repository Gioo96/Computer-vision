#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include "canny_hough.cpp"

using namespace std;
using namespace cv;

// Function headers
void onCannyThreshold_1(int pos, void *userdata);
void onCannyThreshold_2(int pos, void *userdata);
void onHoughLineThetaden(int pos, void *userdata);
void onHoughCircleMinDist(int pos, void *userdata);
Point getIntersection(vector<vector<Point>> line);
vector<Point> pair_line(Vec2f line);

int main(int argc, const char * argv[]) {

    // Load image
    Mat src = imread(argv[1],IMREAD_COLOR);
    if (!src.data) {
        cout<<"Error loading the image"<<endl;
        return -1;
    }
    namedWindow("Input image");
    imshow("Input image", src);
    waitKey(0);
    

    // Find edges (input image of HoughLine)
    Mat dst = src.clone();
    int threshold1_Canny = 800;
    int count_th_Canny = 1500;
    int threshold2_Canny = 500;
    int aperture_size_Canny = 3;
    Canny_edge canny(dst, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
    canny.doAlgorithm();
    
    cout<<"Threshold 1: " + to_string(canny.getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny.getThreshold2())<<endl;
    namedWindow("Canny output (input image of HoughLine)", WINDOW_AUTOSIZE);
    createTrackbar("Threshold 2", "Canny output (input image of HoughLine)", &threshold2_Canny, count_th_Canny);
    createTrackbar("Threshold 1", "Canny output (input image of HoughLine)", &threshold1_Canny, count_th_Canny);
    imshow("Canny output (input image of HoughLine)",canny.getResult());
    createTrackbar("Threshold 1", "Canny output (input image of HoughLine)", &threshold1_Canny, count_th_Canny, onCannyThreshold_1,static_cast<void*>(&canny));
    createTrackbar("Threshold 2", "Canny output (input image of HoughLine)", &threshold2_Canny, count_th_Canny, onCannyThreshold_2,static_cast<void*>(&canny));
    waitKey(0);
    
    // Find lines
    int rho_HoughLine = 1;
    int thetaden_HoughLine = 16; // denominator of theta parameter
    int votes_threshold = 100;
    HoughLine hough_l(canny.getResult(),rho_HoughLine,CV_PI/thetaden_HoughLine,votes_threshold);
    hough_l.doAlgorithm();
    cout<<"Theta: " + to_string(hough_l.getTheta())<<endl;
    namedWindow("HoughLine output", WINDOW_AUTOSIZE);
    createTrackbar("Theta denominator", "HoughLine output", &thetaden_HoughLine, 360);
    imshow("HoughLine output",hough_l.getResult());
    createTrackbar("Theta denominator", "HoughLine output", &thetaden_HoughLine, 360, onHoughLineThetaden,static_cast<void*>(&hough_l));
    imshow("HoughLine output", hough_l.getResult());
    waitKey(0);
    
    // Detected lines (output of the hough line algorithm)
    vector<Vec2f> detected_lines = hough_l.getLines(); // (rho,theta) coordinates
    
    // line1_line2 = [2 pairs of pixel coord that identify line1, 2 pairs of pixel coord that identify line2]
    vector<vector<Point>> line1_line2;
    for( size_t i = 0; i < detected_lines.size(); i++ ) {
        line1_line2.push_back(pair_line(detected_lines.at(i)));
    }
    
    // line3 = [2 pairs of pixel coord that identify line3]
    vector<Point> line3;
    line3.push_back(Point(10,src.rows-1));
    line3.push_back(Point(30,src.rows-1));
    
    // line1_line3 = [2 pairs of pixel coord that identify line1, 2 pairs of pixel coord that identify line3]
    vector<vector<Point>> line1_line3;
    line1_line3.push_back(line1_line2.at(0));
    line1_line3.push_back(line3);
    
    // line2_line2 = [2 pairs of pixel coord that identify line2, 2 pairs of pixel coord that identify line3]
    vector<vector<Point>> line2_line3;
    line2_line3.push_back(line1_line2.at(1));
    line2_line3.push_back(line3);

    // Get intersections
    Point int1 = getIntersection(line1_line2);
    Point int2 = getIntersection(line1_line3);
    Point int3 = getIntersection(line2_line3);

    // Vertices
    vector<Point> vertices;
    vertices.push_back(int1);
    vertices.push_back(int2);
    vertices.push_back(int3);
    
    // Find edges (input image of HoughCircles)
    threshold1_Canny = 1300;
    threshold2_Canny = 500;
    canny.setThreshold1(threshold1_Canny);
    canny.setThreshold2(threshold2_Canny);
    canny.doAlgorithm();
    cout<<"Threshold 1: " + to_string(canny.getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny.getThreshold2())<<endl;
    namedWindow("Canny output (input image of HoughCircles)");
    imshow("Canny output (input image of HoughCircles)", canny.getResult());
    waitKey(0);
    
    // Find circles
    int dp_HoughCircle = 7;
    int minDist_HoughCircle = src.rows;
    HoughCircle hough_c(canny.getResult(), dp_HoughCircle, minDist_HoughCircle);
    hough_c.doAlgorithm();
    namedWindow("HoughCircle output", WINDOW_AUTOSIZE);
    createTrackbar("MinDist", "HoughCircle output", &minDist_HoughCircle, src.rows);
    imshow("HoughCircle output",hough_c.getResult());
    createTrackbar("MinDist", "HoughCircle output", &minDist_HoughCircle, src.rows, onHoughCircleMinDist,static_cast<void*>(&hough_c));
    imshow("HoughCircle output", hough_c.getResult());
    waitKey(0);

    
    
    // Detected circle (output of the hough circle algorithm)
    vector<Vec3f> detected_circle = hough_c.getCircles(); // (x,y,radius) coordinates
    Point center(cvRound(detected_circle[0][0]), cvRound(detected_circle[0][1]));
    int radius = cvRound(detected_circle[0][2]);

    // Color in red the found street, color in green the found circle
    Mat output = src.clone();
    vector<vector<Point>> ppt;
    ppt.push_back(vertices);
    fillPoly(output, ppt, Scalar(0,0,255));
    circle(output, center, radius, Scalar(0,255,0), FILLED, FILLED, 0 );
    namedWindow("Output");
    imshow("Output", output);
    waitKey(0);
    
    return 0;
}

void onCannyThreshold_1(int pos, void *userdata) {

    Canny_edge* canny = static_cast<Canny_edge*>(userdata);
    canny->setThreshold1(pos);
    cout<<"Threshold 1: " + to_string(canny->getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny->getThreshold2())<<endl;
    canny->doAlgorithm();
    imshow("Canny output (input image of HoughLine)", canny->getResult());
}

void onCannyThreshold_2(int pos, void *userdata) {

    Canny_edge* canny = static_cast<Canny_edge*>(userdata);
    canny->setThreshold2(pos);
    cout<<"Threshold 1: " + to_string(canny->getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny->getThreshold2())<<endl;
    canny->doAlgorithm();
    imshow("Canny output (input image of HoughLine)", canny->getResult());
}

void onHoughLineThetaden(int pos, void *userdata) {

    HoughLine* hough_l = static_cast<HoughLine*>(userdata);
    hough_l->setTheta(CV_PI/pos);
    cout<<"Theta: " + to_string(hough_l->getTheta())<<endl;
    hough_l->doAlgorithm();
    imshow("HoughLine output", hough_l->getResult());
}

void onHoughCircleMinDist(int pos, void *userdata) {

    HoughCircle* hough_c = static_cast<HoughCircle*>(userdata);
    hough_c->setMinDist(pos);
    cout<<"MinDist: " + to_string(hough_c->getMinDist())<<endl;
    hough_c->doAlgorithm();
    imshow("HoughCircle output", hough_c->getResult());
}

// Given 2 lines (where 1 line is represented by 2 pairs of pixel coordinates) getIntersection returns the pixel coordinate of the intersection between the 2 lines
Point getIntersection(vector<vector<Point>> line) {
   // First Line represented as ax + by = c
   double a = line.at(0).at(1).y - line.at(0).at(0).y;
   double b = line.at(0).at(0).x - line.at(0).at(1).x;
   double c = a*(line.at(0).at(0).x) + b*(line.at(0).at(0).y);
   // Second line represented as a1x + b1y = c1
   double a1 = line.at(1).at(1).y - line.at(1).at(0).y;
   double b1 = line.at(1).at(0).x - line.at(1).at(1).x;
   double c1 = a1*(line.at(1).at(0).x)+ b1*(line.at(1).at(0).y);
   double det = a*b1 - a1*b;
   if (det == 0) {
      throw "The 2 lines are parallel";
   }
   else {
      double x = (b1*c - b*c1)/det;
      double y = (a*c1 - a1*c)/det;
      return Point(static_cast<int>(x),static_cast<int>(y));
   }
}

// Given the pair (rho,theta) that identifies a line, pair_line gets the 2 pixel coordinates that identify the same line
vector<Point> pair_line(Vec2f line) {
    vector<Point> pair;
    float rho = line[0], theta = line[1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    pair.push_back(pt1);
    pair.push_back(pt2);
    return pair;
}

#include "tracking.cpp"


using namespace std;
using namespace cv;

// Function's declaration
void onMouse(int event, int x, int y, int flags, void* param);
vector<Point2f> sort_corners(vector<Point2f> corners, int cols, int rows);

// Global variables
vector<Point2f> object_corners;
int num_corners = 0;

int main(int argc, const char * argv[]) {

    // Provide 2 arguments
    if (argc != 3) { // Also spaces are included

            perror("Please provide valid data");
            return -1;
    }
    
    String path = argv[1];
    
    // vector of frames
    vector<Mat> frames;
    VideoCapture cap(path);
    if (cap.isOpened()) {
        for(;;) {
            cv::Mat frame;
            cap.read(frame);
            if (frame.empty()) {
                cout<< "Blank frame grabbed\n";
                break;
            }
            
            frames.push_back(frame);
        }
    }
    else {
        
        cout<<"Video has not been captured!"<<endl;
        return -1;
    }

    // Load dataset and video frames
    String dataset_path = argv[2];
    Tracking track(frames, dataset_path);

    vector<Mat> H;
    Mat img_keypoints;
    vector<vector<Point2f>> allcoords_keypoints;
    
    allcoords_keypoints = track.visualizeGoodKeypoints(H, img_keypoints);
    cout<<"Press a key"<<endl;
    
    // Get corners of each object in the images of the dataset
    vector<vector<Point2f>> corners;
    for (int i=0; i<track.images_dataset.size(); i++) {
        
        cout<<"Select the 4 corners of the object"<<endl;
        String window_name = "Object n." + to_string(i+1);
        namedWindow(window_name);
        Mat* image = &track.images_dataset.at(i);
        imshow(window_name, *image);
        setMouseCallback(window_name, onMouse, static_cast<void*>(image));
        waitKey(0);

        // Sort corners
        vector<Point2f> corners_sorted;
        corners_sorted = sort_corners(object_corners, track.images_dataset.at(i).cols, track.images_dataset.at(i).rows);
 
        corners.push_back(corners_sorted);
    }
    
    // Before drawRect, corners will contain the position of the corners of the objects in dataset imgs
    // After drawRect, corners will contain the position of the corners of the objects in frame_0
    track.drawRect(H, img_keypoints, corners, 0);

    // Track objects
    cout<<"---------"<<endl<<"Video in execution"<<endl;
    track.trackObjects(corners, allcoords_keypoints);
    
    return 0;
}

void onMouse(int event, int x, int y, int flags, void* userdata) {

    if (event != EVENT_LBUTTONDOWN) {

        return;
    }
    else {

        if (num_corners == 4) {

            vector<Point2f> init;
            object_corners = init;
            num_corners = 0;
        }

        Point2f corner(x,y);
        object_corners.push_back(corner);
        num_corners ++;

        cout<<"Corner n." + to_string(num_corners) + ":"<<corner<<endl;

        if (num_corners == 4) {

            cout<<"Press a key"<<endl;
        }
    }
}

vector<Point2f> sort_corners(vector<Point2f> corners, int cols, int rows) {

    vector<Point2f> corners_sorted;
    vector<int> j; // j will contain the index of corners to be add at the end of corners_sorted
    int i = 0;

    // Find not correct corners index
    while (i < corners.size()-1) {

        if (abs(corners.at(i).x - corners.at(i+1).x) > cols/2 && abs(corners.at(i).y - corners.at(i+1).y) > rows/2) {
            j.push_back(i+1);
        }
        i ++;
    }

    // If the index of a corner is correct then add the corner to corners_sorted
    for (i=0; i<corners.size(); i++) {

        bool found = false;
        for (int k=0; k<j.size(); k++) {

            if (i == j.at(k)) {

                found = true;
            }
        }
        if (!found) {

            corners_sorted.push_back(corners.at(i));
        }
    }

    // Add all the remaining corners at the end of corners_sorted
    for (i=0; i<j.size(); i++) {

        corners_sorted.push_back(corners.at(j.at(i)));
    }

    return corners_sorted;
}


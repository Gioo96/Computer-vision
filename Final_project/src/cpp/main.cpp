#include "detection_segmentation.cpp"
//#include <fstream>


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    
    // Provide 2 arguments
    if (argc != 5) { // Also spaces are included

            perror("Please provide valid data");
            return -1;
    }

    String pattern_kaggle = argv[1];
    String pattern_venice = argv[2];
    String model_CNN_pb = argv[3];
    String ground_truth_segmentation_path = argv[4];
    
    //
    // KAGGLE DATASET
    //
    cout<<"KAGGLE DATASET"<<endl<<endl<<"DETECTION";
    Detection kaggle_det(pattern_kaggle, 1);

    // Test images on which the performance is needed to be evaluated
    vector<Mat> test_images = kaggle_det.test_images;

    // Detection
    kaggle_det.detection(test_images, model_CNN_pb);

    cout<<"--------------"<<endl<<"--------------"<<endl;

    // Segmentation
    cout<<"SEGMENTATION";
    Segmentation kaggle_seg(pattern_kaggle, 1);
    kaggle_seg.segmentation(ground_truth_segmentation_path);
    

    //
    // VENICE DATASET
    //
    cout<<endl<<"VENICE DATASET"<<endl<<endl<<"DETECTION";
    Detection venice_det(pattern_venice, 0);

    vector<Mat> test_imagess = venice_det.test_images;

    // Equalization on V channel
    vector<Mat> image_equalized;
    for (int i = 0; i<test_images.size(); i++){

        vector<Mat> planes;
        Mat image_equal;
        Mat image_hsv;
        vector<Mat> equalized_channel;

        cvtColor(test_images.at(i), image_hsv, COLOR_BGR2HSV);
        split(image_hsv, planes);
        equalizeHist(planes[2], planes[2]);
        merge(planes,image_equal);
        cvtColor(image_equal, image_equal, COLOR_HSV2BGR);
        image_equalized.push_back(image_equal);
    }

    // Smoothing
    vector<Mat> image_smoothed;
    for (int i = 0; i<test_images.size(); i++){

        Mat image_smooth;
        GaussianBlur(image_equalized.at(i), image_smooth, Size(3,3), 0);
        image_smoothed.push_back(image_smooth);
    }

    // Detection
    venice_det.detection(test_imagess, model_CNN_pb);
   
    // Segmentation
    cout<<"SEGMENTATION";
    Segmentation venice_seg(pattern_venice, 0);
    venice_seg.segmentation(ground_truth_segmentation_path);
    
    return 0;
}

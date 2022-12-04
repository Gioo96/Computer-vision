#include "detection_segmentation.h"

using namespace cv;
using namespace std;


//
// DATASET CLASS
//
// CONSTRUCTOR
Dataset::Dataset(String images_path, String model_CNN_path) {
    
    // Create Boat
    std::__fs::filesystem::create_directories("../data/Boat");
    std::__fs::filesystem::create_directories("../data/No Boat");
    
    int stepSize = 50;
    int windowSize = 300;
    double scale = 1.5;

    // Read images
    vector<String> fn;
    glob(images_path, fn);
    
    // Total number of images
    int tot_numb_images = fn.size();
    
    for (int num_im = 0; num_im < tot_numb_images; num_im ++) {
        
        // Read the image
        Mat im = imread(fn.at(num_im), IMREAD_COLOR);
        
        // Number of No_boats
        int num_noboat = 0;
        
        // Number of Boats
        int num_boat = 0;
        
        while((im.rows >= windowSize) && (im.cols >= windowSize)) {

            // Sliding window
            int i = 0; // row

            int count_slidWindow = 0;
            while (i <= im.rows - windowSize) {

                for (int j = 0; j <= im.cols - windowSize; j += stepSize) {

                    Rect rect(j, i, windowSize, windowSize);
                    Mat sub_im = im.clone();
                    sub_im = sub_im(rect); // Sub-image

                    Mat draw_window = im.clone();
                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);

                    // Save Boat-No_Boat images in the corresponding local directories
                    if (count_slidWindow % 2 == 0) {

                        // CNN prediction
                        dnn::Net net = dnn::readNetFromTensorflow(model_CNN_path);
                        Mat img_toNet_blob = cv::dnn::blobFromImage(sub_im);
                        net.setInput(img_toNet_blob);
                        Mat res;
                        net.forward(res);

                        // sub_im is detected as Boat
                        if (round(res.at<float>(0)) == 1) {

                            
                            imwrite("../data/Boat/img" + to_string(num_im) + "_" + to_string(num_boat) + ".jpg", sub_im);
                            num_boat ++;
                        }

                        else {

                            imwrite("../data/No Boat/img" + to_string(num_im) + "_" + to_string(num_noboat) + ".jpg", sub_im);
                            num_noboat ++;
                        }
                    }
                    
                    count_slidWindow ++;
                }
    
                i += 80;
            }
            
            Size size(im.cols/scale, im.rows/scale);
            resize(im, im, size);
        }
    }
}


//
// BASE CLASS
//
// CONSTRUCTOR
Base::Base(String pattern, int fl) {

    flag = fl;
    if (flag == 1) {
    
        pattern = pattern + "/*.jpg";
    }
    else {
        
        pattern = pattern + "/*.png";
    }
    
    vector<String> fn;
    
    try {
        glob(pattern, fn, false);
    }
    catch (exception& ex) {
        cout<<"Invalid pattern name"<<endl;
    }

    if (!fn.size()) {
        
        cout<<"Error loading the images"<<endl;
    }
    else {
        
        int count = fn.size(); // Number of images
        for (int i = 0; i < count; i++) {
            
            test_images.push_back(imread(fn[i], IMREAD_COLOR));
            namedWindow("Image");
        }
    }
}


//
// DETECTION CLASS
//
// CONSTRUCTOR
Detection::Detection(String pattern, int fl) : Base(pattern, fl) {

}
    
// METHODS
// detection
void Detection::detection(vector<Mat> image, String model_CNN_pb) {

    int scaling;
    int min_rect;
    double eps;
    int stepSize;
    int windowSize_rows;
    int windowSize_cols;

    if (flag == 1) {

        scaling = 1;
        min_rect = 1;
        eps = 0.8;
        stepSize = 40;
        windowSize_rows = 115;
        windowSize_cols = 115;
    }

    else {

        scaling = 2;
        min_rect = 2;
        eps = 0.6;
        stepSize = 30;
        windowSize_rows = 150;
        windowSize_cols = 150;
    }

    for (int index_image = 0; index_image < image.size(); index_image++) {

        cout<<endl<<"--------------"<<endl;
        cout<<"Boat detection on image N. "<<to_string(index_image)<<"..."<<endl;
        int tot_num_boats_08 = 0; // Total number of detected boats (counted if prob >= 0.8), where prob is the probability for the sub-image to belong to the class "Boat" given by the CNN

        int tot_num_boats_05 = 0; // Total number of detected boats (counted if prob >= 0.5), where prob is the probability for the sub-image to belong to the class "Boat" given by the CNN


        Mat image_final = image.at(index_image).clone();
        
        vector<Rect> temp_rects;

        vector<Rect> rects_08;
        vector<Rect> rects_05;
        int merged = 0;

        // Sliding window : only the first level of the pyramid is considered
        int i = 0; // row

        while (i <= image.at(index_image).rows - windowSize_rows) {

            for (int j = 0; j <= image.at(index_image).cols - windowSize_cols; j += stepSize) { // col

                Rect rect(j, i, windowSize_cols, windowSize_rows);
                Mat draw_window = image.at(index_image).clone(); // Image in which the sliding windows are drawn
                Mat window = draw_window(rect); // Sub-image to be given to the CNN for classification

                // Resize window (it has to be 300x300 since CNN was trained with images with such sizes)
                resize(window, window, Size(300, 300));

                // Load CNN model
                dnn::Net net = dnn::readNetFromTensorflow(model_CNN_pb);
                Mat img_toNet_blob = cv::dnn::blobFromImage(window);
                net.setInput(img_toNet_blob);
                Mat prob;
                net.forward(prob);

                // Probability >= 0.8
                if (prob.at<float>(0) >= 0.8) {

                    rects_08.push_back(rect);
                    tot_num_boats_08 ++;
                }

                else {

                    if (round(prob.at<float>(0)) == 1) {


                        rects_05.push_back(rect);
                    }
                }

                rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 3);
                imshow("Boat detection on image N. " + to_string(index_image), draw_window);
                waitKey(1);
            }

            i += stepSize;
        }

        if (tot_num_boats_08 == 0 && tot_num_boats_05 > 0) {

            for (int i = 0; i < rects_05.size(); i ++) {

                Rect rect_temporal = rects_05.at(i);
                String boat = "BOAT N.";
                putText(image_final, boat + to_string(i+1) , Point(rect_temporal.x, rect_temporal.y - 5),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                rectangle(image_final, rects_05.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
            }
        }

        else if (tot_num_boats_08 > 0) {

            temp_rects = rects_08;

            if (rects_08.size() > 1) {

                // rect_08 now contains the seed rectangles
                groupRectangles(rects_08, min_rect, eps);
            }

            vector<Rect> output_rect = rect_return(temp_rects, rects_08, scaling);

            vector<Rect> output_first_rect = output_rect;

            // INTERSECTION
            int equal = 2;
            int x_inter;
            int y_inter;
            int height_inter;
            int width_inter;
            double area_inter;
            double area_1;
            double area_2;
            double area_max;
            vector<int> rect_saved;

            // Verify how intersect
            for (int i = 0; i < output_first_rect.size(); i ++){

                int x_current_1 = output_first_rect.at(i).x;
                int y_current_1 = output_first_rect.at(i).y;
                int height_current_1 = output_first_rect.at(i).height;
                int width_current_1 = output_first_rect.at(i).width;

                for (int j = 0; j<output_first_rect.size(); j ++){

                    if (j!= i){

                        int x_current_2 = output_first_rect.at(j).x;
                        int y_current_2 = output_first_rect.at(j).y;
                        int height_current_2 = output_first_rect.at(j).height;
                        int width_current_2 = output_first_rect.at(j).width;

                        x_inter = max(x_current_1, x_current_2);
                        y_inter = max(y_current_1, y_current_2);
                        height_inter = min(y_current_1+height_current_1, y_current_2+height_current_2) - y_inter;
                        width_inter = min(x_current_1+width_current_1, x_current_2+width_current_2) - x_inter;

                        if (height_inter >0 && width_inter>0){

                            area_inter = height_inter * width_inter;
                            area_1 = height_current_1 * width_current_1;
                            area_2 = height_current_2 * width_current_2;

                            area_max = max(area_1, area_2);

                            double eval = area_inter/area_max;

                            if (eval > 0.1){


                                if (area_1 >= area_2){

                                    rect_saved.push_back(i);
                                    equal =1;
                                }
                                else {

                                    rect_saved.push_back(j);
                                    equal = 1;

                                }
                            }
                        }
                    }
                }
            }

            if (equal == 1) {

                vector<Rect> output_reect;
                output_reect.push_back(output_first_rect.at(rect_saved[0]));
                getMetrics(output_reect, image_final, index_image);

                Rect rect_temporal = output_first_rect.at(rect_saved[0]);
                String boat = "BOAT N.";
                putText(image_final, boat + to_string(1), Point(rect_temporal.x, rect_temporal.y - 5), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                rectangle(image_final, output_first_rect.at(rect_saved[0]), Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
            }

            else if (equal == 0) {

                for (int i = 0; i < rect_saved.size(); i ++) {

                    Rect rect_temporal = output_first_rect.at(i);
                    String boat = "BOAT N.";
                    putText(image_final, boat + to_string(i + 1), Point(rect_temporal.x, rect_temporal.y - 5), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                    rectangle(image_final, output_first_rect.at(rect_saved[i]), Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
                }
            }

            else if (equal == 2) {

                getMetrics(output_first_rect, image_final, index_image);

                for (int i = 0; i < output_first_rect.size(); i ++) {

                    Rect rect_temporal = output_first_rect.at(i);
                    String boat = "BOAT N.";
                    putText(image_final, boat + to_string(i + 1), Point(rect_temporal.x, rect_temporal.y - 5), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                    rectangle(image_final, output_first_rect.at(i), Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
                }
            }
        }

        imshow("Detected boats N."+to_string(index_image), image_final);
        cout<<"Please press a key"<<endl;
        waitKey(0);
        destroyAllWindows();
    }
}


// rect_return
vector<Rect> Detection::rect_return(vector<Rect> all_rects, vector<Rect> seed_rects, int scaling) {
    
    int distance;
    int change;
    
    // scaling identifies how interesection is needed   
    vector<Rect> output_rect;
    vector<int> tot_x;
    vector<int> tot_y;
    vector<int> tot_height;
    vector<int> tot_width;
    
    for (int k = 0; k < seed_rects.size(); k ++) {
        
        change = 0;
        
        Rect current_rect = seed_rects.at(k);
        int x_current = current_rect.x;
        int y_current = current_rect.y;
        int height_current = current_rect.height;
        int width_current = current_rect.width;
        
        for (int h = 0; h < all_rects.size(); h ++) {
            
            Rect current_all_rect = all_rects.at(h);
            int x_all_current = current_all_rect.x;
            int y_all_current = current_all_rect.y;
            int height_all_current = current_all_rect.height;
            int width_all_current = current_all_rect.width;
            
            
            // MORE RIGHT
            if (x_all_current <= x_current+width_current/scaling && x_all_current >= x_current && x_all_current+width_all_current>= x_current+width_current) {
                
                if (y_all_current <= y_current+height_current/scaling && y_all_current >= y_current) {
                    
                    //y_all_current+height_all_current>y_current+height_current
                    
                    height_current = height_current + y_current;
                    width_current = width_current + (x_all_current+width_all_current-x_current-width_current);
                    height_current = height_current + (y_all_current+height_all_current-y_current-height_current);
                    change ++;
                }
                
                else if (y_all_current+height_all_current/scaling>=y_current && y_all_current<=y_current) {
                    
                    width_current = width_current + (x_all_current+width_all_current-x_current-width_current);
                    y_current = y_all_current;
                    change ++;
                }
                    
            }
            
            // MORE LEFT
            else if (x_all_current+width_all_current/scaling>=x_current && x_all_current<=x_current) {
                
                // MORE DOWN
                if (y_all_current <= y_current+height_current/scaling && y_all_current >= y_current) {
                    
                    //y_all_current+height_all_current>y_current+height_current
                    
                    width_current = width_current + x_current-x_all_current;
                    x_current = x_all_current;
                    height_current = y_all_current+height_all_current-y_current;
                    change ++;
                    }
                
                // MORE UP
                else if (y_all_current+height_all_current/scaling>=y_current && y_all_current<=y_current) {
                    
                    height_current = y_current-y_all_current+height_current;
                    y_current = y_all_current;
                    width_current = x_current-x_all_current + width_current;
                    x_current = x_all_current;
                    change ++;
                }
            }
        }
        
        Rect alone_output_rect = Rect(x_current, y_current, width_current, height_current);
        output_rect.push_back(alone_output_rect);
    }

    return output_rect;
}

// getMetrics
void Detection::getMetrics(vector<Rect> predicted_rects, Mat image_predicted, int index) {
    
    int length_file = test_images.size();

    vector<vector<Rect>> ground_truth;
    vector<Rect> image_0;
    vector<Rect> image_1;
    vector<Rect> image_2;
    vector<Rect> image_3;
    vector<Rect> image_4;
    vector<Rect> image_5;
    vector<Rect> image_6;
    vector<Rect> image_7;
    vector<Rect> image_8;
    vector<Rect> image_9;
    vector<Rect> image_10;
    vector<Rect> image_11;
    
    if (length_file == 10) {
        
        image_0.push_back(Rect(280,145,810,600));
        image_0.push_back(Rect(400,620,280,170));
        image_0.push_back(Rect(1100,650,130,120));
        
        image_1.push_back(Rect(720,420,120,150));
        
        image_2.push_back(Rect(720,220,250,290));
        
        image_3.push_back(Rect(310,430,640,250));
        
        image_4.push_back(Rect(780,620,110,110));
        
        image_5.push_back(Rect(80,70,1000,370));
        
        image_6.push_back(Rect(220,410,280,140));
        image_6.push_back(Rect(780,400,300,130));
        
        image_7.push_back(Rect(230,300,180,100));
        image_7.push_back(Rect(560,300,180,100));
        image_7.push_back(Rect(910,300,150,80));
        
        image_8.push_back(Rect(560,440,370,300));
        
        image_9.push_back(Rect(295,375,430,183));

        ground_truth.push_back(image_0);
        ground_truth.push_back(image_1);
        ground_truth.push_back(image_2);
        ground_truth.push_back(image_3);
        ground_truth.push_back(image_4);
        ground_truth.push_back(image_5);
        ground_truth.push_back(image_6);
        ground_truth.push_back(image_7);
        ground_truth.push_back(image_8);
        ground_truth.push_back(image_9);
    }
    
    else if (length_file == 12) {
        
        image_0.push_back(Rect(950,100,100,150));
        image_0.push_back(Rect(10,280,260,160));
        image_0.push_back(Rect(150,460,380,300));
        
        image_1.push_back(Rect(500,280,130,100));
        image_1.push_back(Rect(640,80,150,110));
        image_1.push_back(Rect(310,290,140,140));
        image_1.push_back(Rect(460,1,34,25));
        
        image_2.push_back(Rect(300,300,150,130));
        image_2.push_back(Rect(750,120,140,100));
        
        image_3.push_back(Rect(950,200,190,330));
        image_3.push_back(Rect(600,580,290,300));
        
        image_4.push_back(Rect(950,100,180,100));
        image_4.push_back(Rect(500,700,450,200));
        image_4.push_back(Rect(1,700,400,220));
        image_4.push_back(Rect(90,250,200,100));
        image_4.push_back(Rect(400,250,450,150));
        
        image_5.push_back(Rect(870,80,100,100));
        image_5.push_back(Rect(300,120,320,100));
        image_5.push_back(Rect(100,750,600,200));
        image_5.push_back(Rect(300,570,500,100));
        image_5.push_back(Rect(380,500,400,110));
        image_5.push_back(Rect(430,420,400,100));
        image_5.push_back(Rect(470,400,350,80));
        image_5.push_back(Rect(950,150,150,80));
        image_5.push_back(Rect(500,80,100,50));
        
        image_6.push_back(Rect(100,500,200,450));
        image_6.push_back(Rect(500,500,200,450));
        image_6.push_back(Rect(250,100,450,150));
        image_6.push_back(Rect(800,450,250,400));
        image_6.push_back(Rect(980,450,220,330));
        
        image_7.push_back(Rect(700,320,200,150));
        image_7.push_back(Rect(850,500,250,100));
        image_7.push_back(Rect(950,400,180,100));
        image_7.push_back(Rect(1000,300,190,100));

        image_8.push_back(Rect(650,550,200,300));
        image_8.push_back(Rect(250,350,220,150));
        image_8.push_back(Rect(50,350,150,80));
        image_8.push_back(Rect(30,300,260,80));
        image_8.push_back(Rect(820,100,190,120));
        
        image_9.push_back(Rect(1,800,250,100));
        image_9.push_back(Rect(170,320,750,200));
        image_9.push_back(Rect(530,730,180,250));
        image_9.push_back(Rect(620,570,240,360));
        image_9.push_back(Rect(750,550,250,350));
        
        image_10.push_back(Rect(630,700,450,150));
        image_10.push_back(Rect(1,720,400,200));
        image_10.push_back(Rect(1,630,330,150));
        image_10.push_back(Rect(300,230,600,170));
        image_10.push_back(Rect(980,130,100,80));
        image_10.push_back(Rect(960,100,200,100));
        
        image_11.push_back(Rect(230,580,450,150));
        image_11.push_back(Rect(850,520,350,150));
        image_11.push_back(Rect(580,400,300,100));

        ground_truth.push_back(image_0);
        ground_truth.push_back(image_1);
        ground_truth.push_back(image_2);
        ground_truth.push_back(image_3);
        ground_truth.push_back(image_4);
        ground_truth.push_back(image_5);
        ground_truth.push_back(image_6);
        ground_truth.push_back(image_7);
        ground_truth.push_back(image_8);
        ground_truth.push_back(image_9);
        ground_truth.push_back(image_10);
        ground_truth.push_back(image_11);
    }
    
    
    
    vector<Rect> current_image = ground_truth.at(index);
    
    int size_int = min(current_image.size(), predicted_rects.size());
    
    for (int i = 0; i < current_image.size(); i++) {
        
        //ground_Truth.push_back(current_image.at(i));
        rectangle(image_predicted, current_image.at(i), Scalar(255,255,255), 3);
    }
    
    // TRUE POSITIVES
    vector<int> true_pos;
    
    cout << "METRICS OF IMAGE "<<to_string(index)<<endl;
    
    for (int j = 0; j < current_image.size(); j++) {
        
        for (int i = 0; i < predicted_rects.size(); i++){
            
            // EVALUATION OF THE METRICS IoU
            int x_truth = current_image.at(j).x;
            int y_truth = current_image.at(j).y;
            int width_truth = current_image.at(j).width;
            int height_truth = current_image.at(j).height;
            double area_truth = width_truth * height_truth;
            
            int x_pred = predicted_rects.at(i).x;
            int y_pred = predicted_rects.at(i).y;
            int width_pred = predicted_rects.at(i).width;
            int height_pred = predicted_rects.at(i).height;
            double area_pred = width_pred * height_pred;
            
            int x_inter = max(x_pred, x_truth);
            int y_inter = max(y_pred, y_truth);
            //cout << "Y inter "<< y_inter << endl;
            int height_inter = min(y_pred + height_pred, y_truth + height_truth) - y_inter;
            int width_inter = min(x_pred + width_pred, x_truth + width_truth) - x_inter;
            
            // Intersection
            double area_intersected = height_inter * width_inter;
    
            // Union
            double area_union = area_pred + area_truth - area_intersected;
            
            // IoU metrics
            double IoU_metrics = area_intersected/area_union;
            
            if (IoU_metrics > 0.09 && height_inter > 0 && width_inter > 0) {
                
                cout<<"BOAT N." << to_string(i+1) <<": " << IoU_metrics << endl;
                true_pos.push_back(i+1);
            }
            
        }
    }
    
    // FALSE POSITIVES
    vector<int> false_pos;
    vector<int>::iterator it;
    
    for (int i = 0; i < predicted_rects.size(); i++){
        
        it = find(true_pos.begin(), true_pos.end(), i+1);
        if (it == true_pos.end()) {

            cout << "The rectangle N. "<< i+1<< " is a false positive"<< endl;
        }
    }
    
    for (int k = 0; k < false_pos.size(); k ++) {
        
        cout<< "The rectangle N. " << false_pos.at(k) << " is a false positive" << endl;
    }
}


//
// SEGMENTATION CLASS
//
// CONSTRUCTOR
Segmentation::Segmentation(String pattern, int fl) : Base(pattern, fl) {

}

// METHODS
// segmentation
void Segmentation::segmentation(String ground_truth_segmentation_path) {
    
    // KAGGLE DATASET
    if (flag == 1) {
        
        for (int i = 0; i < test_images.size(); i ++) {
            
            cout<<endl<<"--------------"<<endl;
            cout<<"Sea segmentation on image N. "<<to_string(i)<<"..."<<endl;
            Mat image_test = test_images.at(i).clone();
            if (i != 9) {
                
                // Smooth the image
                Mat image_smoothed;
                GaussianBlur(image_test, image_smoothed, Size (27,27), 0, 0);
                
                // KMEANS SEGMENTATION BASED ON COLOR
                // Covert the image into CV_32F
                Mat data;
                image_smoothed.convertTo(data, CV_32F);
                
                // Reshape data
                data = data.reshape(1, data.total());
              
                // Outputs of Kmeans
                Mat labels;
                Mat centers;
                kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);

                // color_segmented_img has size (rows*cols, 1), channels = 3
                Mat color_segmented_img = data.reshape(3,data.rows);
                
                for (int j = 0; j < labels.rows; j ++) {
                 
                    int center_id = labels.at<int>(j);
                    color_segmented_img.at<Vec3f>(j) = centers.at<Vec3f>(center_id);
                }
                
                // color_segmented_img has size (rows, cols), channels = 3
                color_segmented_img = color_segmented_img.reshape(3, test_images.at(i).rows);
                
                // Convert color_segmented_img to CV_8UC3
                color_segmented_img.convertTo(color_segmented_img, CV_8UC3);
                
                // CANNY EDGE
                // Find edges (input image of HoughLine)
                int threshold1_Canny = 90;
                int count_th_Canny = 1000;
                int threshold2_Canny = 155;
                int aperture_size_Canny = 3;
                
                Mat edges;
                Canny(color_segmented_img, edges, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
                
                // HOUGH LINE
                // Find line which delimits the sea
                int rho_HoughLine = 1;
                int thetaden_HoughLine = 50; // denominator of theta parameter
                int votes_threshold = 100;
                
                HoughLine hough_l(edges, rho_HoughLine, CV_PI/thetaden_HoughLine, votes_threshold);
                hough_l.doAlgorithm();
                
                // Corrdinate pixels of the pair of points foreach line
                vector<pair<Point, Point>> lines = hough_l.getLines();
                
                // Result image after dilation
                Mat dst_dilate_above;
                
                // Result image after erosion+dilation
                Mat dst_erode_dilate_below;
                
                // Find max .y
                int current_y = max(lines.at(0).first.y, lines.at(0).second.y);
                for (int k = 1; k < lines.size(); k ++) {
                    
                    current_y = max(current_y, lines.at(k).first.y);
                    current_y = max(current_y, lines.at(k).second.y);
                }
                
                // DILATE ABOVE
                // Sub-image to dilate
                Mat dilate_subimg = color_segmented_img(Rect(0, 0, color_segmented_img.cols, current_y));
                
                // Structuring element
                Mat str_el_dilate = getStructuringElement(MORPH_RECT, Size(21,21));
                dilate(dilate_subimg, dst_dilate_above, str_el_dilate, Point(-1,-1), 15);
                
                // ERODE+DILATE BELOW
                // Sub-image to erode
                Mat erode_subimg = color_segmented_img(Rect(0, current_y, color_segmented_img.cols, color_segmented_img.rows - current_y));
                
                // Structuring element
                Mat str_el_erode_dilate = getStructuringElement(MORPH_RECT, Size(13,13));
                erode(erode_subimg, dst_erode_dilate_below, str_el_erode_dilate, Point(-1,-1), 2);
                dilate(dst_erode_dilate_below, dst_erode_dilate_below, str_el_erode_dilate, Point(-1,-1), 2);
                
                // Resulting image
                Mat dst_erode_dilate = color_segmented_img.clone();
                vconcat(dst_dilate_above, dst_erode_dilate_below, dst_erode_dilate);
                
                Mat original_segmentated;
                hconcat(image_test, dst_erode_dilate, original_segmentated);
                
                imshow("Sea segmentation on image N. " + to_string(i), original_segmentated);
                
                cvtColor(dst_erode_dilate, dst_erode_dilate, COLOR_BGR2GRAY);
                
                // White: sea, Black: not sea
                swap_colors(dst_erode_dilate);
                
                //segmented_images.push_back(dst_erode_dilate);
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, dst_erode_dilate, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
            }
            
            // image.at(9) KAGGLE: OTSU
            else {
                
                // Convert to gray scale image
                Mat gray;
                cvtColor(image_test, gray, COLOR_BGR2GRAY);
                
                // Smoothing
                Mat gray_smoothed;
                GaussianBlur(gray, gray_smoothed, Size(21,21), 0);
                
                // Threshold : Otsu
                Mat gray_smoothed_otsu;
                double thresh = 0;
                double maxval = 255;
                threshold(gray_smoothed, gray_smoothed_otsu, thresh, maxval, THRESH_OTSU);
                
                // White: sea, Black: not sea
                swap_colors(gray_smoothed_otsu);
                
                // Show segmentation result
                Mat show_img;
                
                cvtColor(gray_smoothed_otsu, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                //segmented_images.push_back(gray_smoothed_otsu);
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, gray_smoothed_otsu, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
            }
        }
    }
    
    // VENICE DATASET
    else {
        
        for (int i = 0; i < test_images.size(); i ++) {
            
            cout<<endl<<"--------------"<<endl;
            cout<<"Sea segmentation on image N. "<<to_string(i)<<"..."<<endl;
            Mat image_test = test_images.at(i).clone();
            
            // Conversion to HSV
            Mat image_hsv;
            cvtColor(image_test, image_hsv, COLOR_BGR2HSV);

            // Equalization on HSV
            vector<Mat> planes;
            Mat image_equal;
            split(image_hsv, planes);
            equalizeHist(planes[2], planes[2]);
            merge(planes,image_equal);
            cvtColor(image_equal, image_equal, COLOR_HSV2BGR);
            GaussianBlur(image_equal, image_equal, Size(5,5), 0);

            // OTSU
            Mat gray_image;
            cvtColor(image_equal, gray_image, COLOR_BGR2GRAY);
            Mat image_segmented;
            threshold(gray_image, image_segmented, 0, 255, THRESH_OTSU);
            
            if (i == 0) {

                // Erosion
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 5);
                
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 4);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img N. " + to_string(i), original_segmentated);

                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
                
                //imwrite("/Users/gioel/Desktop/images/ven"+to_string(i)+"_seg.jpg", original_segmentated);
            }
            else if (i == 1) {
                
                // Concatenate the original image and segmented image
                Mat show_img;
                swap_colors(image_segmented);
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                imwrite("Users/gioel/Desktop/img1.jpg", original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
                
                imwrite("/Users/gioel/Desktop/img_segmented"+to_string(i)+".jpg", original_segmentated);
            }
            else if (i == 2) {
                
                // Concatenate the original image and segmented image
                Mat show_img;
                swap_colors(image_segmented);
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                imwrite("Users/gioel/Desktop/img2.jpg", original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
                
                imwrite("/Users/gioel/Desktop/img_segmented"+to_string(i)+".jpg", original_segmentated);
            }
            
            else if (i == 3 || i == 11) {
           
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 5);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                swap_colors(image_segmented);
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
            }
            
            else if (i == 4) {

                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 5);
                
                // Erosion
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 5);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
            }
            
            else if (i == 5) {
                
                // Erosion
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 6);
                
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 4);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                //imwrite("/Users/davideallegro/Documents/Università/Laurea magistrale/Computer Vision/Project/Ground_truth/Segmentation/img_segmented" + to_string(i)+ ".jpg", original_segmentated);
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
            }
            else if (i == 6 || i == 7){
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 4);
                
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 3);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
            }
            else if (i == 8 ){
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 5);
                
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 4);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
        }
            else if (i == 9 ){
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 5);
                
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 4);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
        }
            else if (i == 10 ) {
                
                Mat kernel_erode = getStructuringElement(0, Size(7,7));
                erode(image_segmented, image_segmented, kernel_erode, Point(-1,-1), 3);
                
                // Dilation
                Mat kernel_dilate = getStructuringElement(0, Size(7,7));
                dilate(image_segmented, image_segmented, kernel_dilate, Point(-1,-1), 6);
                
                // Concatenate the original image and segmented image
                Mat show_img;
                cvtColor(image_segmented, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                
                // Show result
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                
                // Metric
                double pixel_accuracy = getMetrics(ground_truth_segmentation_path, image_segmented, i);
                
                cout<<"METRIC OF IMAGE "<<to_string(i)<<": "<<to_string(pixel_accuracy)<<endl;
                cout<<"Pleas press a key"<<endl;
                waitKey(0);
        }
        }
    }
}

// pixel_accuracy
double Segmentation::getMetrics(String ground_truth_segmentation_path, Mat segmentated_image, int index) {
    
    // Kaggle
    if (flag == 1) {
        
        ground_truth_segmentation_path += "/Kaggle/*.jpg";
    }
    
    else {
        
        ground_truth_segmentation_path += "/Venice/*jpg";
    }
    
    vector<String> fn;
    glob(ground_truth_segmentation_path, fn);
    vector<Mat> Kaggle_gt; // Ground truth Kaggle images
    
    for (int i = 0; i < fn.size(); i ++) {
        
        Kaggle_gt.push_back(imread(fn.at(i)));
    }
    
    vector<double> accuracy_images;
    
    // Evaluate pixel acccuracy for the given segmented image
        
        cvtColor(segmentated_image, segmentated_image, COLOR_GRAY2BGR);
        double intersection_pixels = 0;
        double union_pixels = 0;
        
        for (int r = 0; r < Kaggle_gt.at(index).rows; r ++) {
            
            for (int c = 0; c < Kaggle_gt.at(index).cols; c ++) {
                
                if (segmentated_image.at<Vec3b>(r,c) == Kaggle_gt.at(index).at<Vec3b>(r,c) && segmentated_image.at<Vec3b>(r,c) == Vec3b(255, 255, 255)) {
                    
                    intersection_pixels ++;
                    union_pixels ++;
                }
                
                else if (segmentated_image.at<Vec3b>(r,c) == Vec3b(255, 255, 255) || Kaggle_gt.at(index).at<Vec3b>(r,c) == Vec3b(255, 255, 255)) {
                    
                    union_pixels ++;
                }
            }
        }
        
        // Evaluate metric
        double pixel_accuracy = intersection_pixels / union_pixels;
    
    return pixel_accuracy;
}

// swap_colors
void Segmentation::swap_colors(Mat& image) {
    
    // Sea: white, Not sea: black
    int count = 1; // count number of classes
    int c1 = static_cast<int>(image.at<uchar>(0,0));
    int c2;
    while (count < 2) {
        for (int r = 0; r < image.rows; r ++) {
            
            for (int c = 0; c < image.cols; c ++) {
                
                if (image.at<uchar>(r,c) != c1) {
                    
                    c2 = static_cast<int>(image.at<uchar>(r,c));
                    count ++;
                }
            }
        }
    }
    
    uchar color_light, color_dark;
    if (c1 > c2) {
        
        color_light = static_cast<uchar>(c2);
        color_dark = static_cast<uchar>(c1);
    }
    else {
        
        color_light = static_cast<uchar>(c1);
        color_dark = static_cast<uchar>(c2);
    }
    
    for (int r = 0; r < image.rows; r ++) {
        
        for (int c = 0; c < image.cols; c ++) {
            
            if (image.at<uchar>(r,c) == color_light) {
                
                image.at<uchar>(r,c) = 255;
            }
            else {
                
                image.at<uchar>(r,c) = 0;
            }
        }
    }
}

// click
void Segmentation::click(Mat image) {

    Mat mask = image.clone();
    
    int count = 0;
    for (int i=0; i<image.rows; i++) {
        
        for (int j=0; j<image.cols; j++) {
            
            if (image.at<Vec3b>(i,j) == Vec3b(127,255,212)) {

                count ++;
            }
        }
    }
    
    //mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
    //Mat image_ex = image;
    imshow("IMAGE", image);
    setMouseCallback("IMAGE", onMouse, static_cast<void*>(&mask));
    waitKey(0);
   
    for (int i=0; i<mask.cols; i++) {
        
        for (int j=0; j<mask.rows; j++) {
            
            if (mask.at<Vec3b>(j,i) == Vec3b(127,255,212)) {
                
                // Sea pixels: white
                mask.at<Vec3b>(j,i) = Vec3b(255,255,255);
            }
            else {
                
                // Not sea pixels: black
                mask.at<Vec3b>(j,i) = Vec3b(0,0,0);
            }
        }
    }
    imshow("Ground truth", mask);
    waitKey(0);
}

// onMouse
void Segmentation::onMouse(int event, int x, int y, int f, void* userdata) {
    
    Mat image = *static_cast<Mat*>(userdata);
    Point point;
    
    if (event == EVENT_LBUTTONDOWN && click_pair < 2) {

        point.x = x;
        point.y = y;
        click_pair ++;
        cout<<"Point: "<<point<<endl;
        //vec_points.push_back(point);
        if (click_pair == 1) {
            
            p1 = point;
        }
        else if (click_pair == 2) {
            
            p2 = point;
        }
    }
    
    if (click_pair == 2) {
        
        p1_p2.first = p1;
        p1_p2.second = p2;
        vec_pairs.push_back(p1_p2);
        cout<<"First: "<<p1_p2.first<<"Second: "<<p1_p2.second<<endl;
        rectangle(image, p1_p2.first, p1_p2.second, Scalar(127,255,212), FILLED);
        click_pair = 0;
    }
    
    imshow("IMAGE", image);
    waitKey(1);
}


//
// HOUGHLINE CLASS
//
// CONSTRUCTOR
HoughLine::HoughLine(cv::Mat input_img, int rho, double theta, int threshold){
    
    input_image_hough = input_img;
    rho_HoughLine = rho;
    theta_HoughLine = theta;
    threshold_HoughLine = threshold;
}

// METHODS
// doAlgorithm
void HoughLine::doAlgorithm() {
    
    vector<Vec2f> lines_theta_rho;
    cv::HoughLines(input_image_hough, lines_theta_rho, rho_HoughLine, theta_HoughLine, threshold_HoughLine);
    Mat cdst;
    cv::cvtColor(input_image_hough, cdst, COLOR_GRAY2BGR);
    vector<pair<Point, Point>> all_point_couple;
    for( size_t i = 0; i < lines_theta_rho.size(); i++ ) {
        
            float rho = lines_theta_rho[i][0], theta = lines_theta_rho[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            Point point1_x_y(pt1.x, pt1.y);
            Point point2_x_y(pt2.x, pt2.y);
    
            pair<Point, Point> point_couple;
            drawStraightLine(cdst, point1_x_y, point2_x_y, Scalar(0, 255, 0));
            
            point_couple.first = point1_x_y;
            point_couple.second = point2_x_y;
            all_point_couple.push_back(point_couple);
        }
    lines = all_point_couple;
    result_image_hough = cdst;
}

// setRho
void HoughLine::setRho(int rho) {
    
    rho_HoughLine = rho;
}

// setTheta
void HoughLine::setTheta(double theta) {
    
    theta_HoughLine = theta;
}

// setThreshold
void HoughLine::setThreshold(int threshold) {
    
    threshold_HoughLine = threshold;
}

// getRho
int HoughLine::getRho() {
    
    return rho_HoughLine;
}


// getTheta
double HoughLine::getTheta() {
    
    return theta_HoughLine;
}

// getThreshold
int HoughLine::getThreshold() {
    
    return threshold_HoughLine;
}

// getLines
std::vector<pair<Point, Point>> HoughLine::getLines() {
    
    return lines;
}

// getResult
cv::Mat HoughLine::getResult() {
    
    return result_image_hough;
}

// drawStraightLine
void HoughLine::drawStraightLine(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Scalar color) {
    
        Point p, q;
        // Check if the line is a vertical line because vertical lines don't have slope
        if (p1.x != p2.x)
        {
                p.x = 0;
                q.x = img.cols;
                // Slope equation (y1 - y2) / (x1 - x2)
                float m = (p1.y - p2.y) / (p1.x - p2.x);
                // Line equation:  y = mx + b
                float b = p1.y - (m * p1.x);
                p.y = m * p.x + b;
                q.y = m * q.x + b;
        }
        else
        {
                p.x = q.x = p2.x;
                p.y = 0;
                q.y = img.rows;
        }

        cv::line(img, p, q, color, 4);
}

// onHoughLineThetaden
void HoughLine::onHoughLineThetaden(int pos, void *userdata) {

    HoughLine* hough_l = static_cast<HoughLine*>(userdata);
    hough_l->setTheta(CV_PI/pos);
    cout<<"Theta: " + to_string(hough_l->getTheta())<<endl;
    cout<<"Theta_den: " + to_string(CV_PI/hough_l->getTheta())<<endl;
    hough_l->doAlgorithm();
    imshow("HoughLine output", hough_l->getResult());
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


using namespace cv;
using namespace std;


void energy_function(Mat &image, Mat &output){
    Mat dx, dy;
    Sobel(image, dx, CV_64F, 1, 0);
    Sobel(image, dy, CV_64F, 0, 1);
    magnitude(dx, dy, output);

    double min_value, max_value, Z;
    minMaxLoc(output, &min_value, &max_value);
    Z = 1/max_value * 255;
    output = output * Z; // Normalize
    output.convertTo(output, CV_8U);
}


int* find_seam(Mat &image){
    int H = image.rows, W = image.cols;

    int dp[H][W];
    for(int c = 0; c < W; c++){
        dp[0][c] = (int)image.at<uchar>(0, c);
    }

    for(int r = 1; r < H; r++){
        for(int c = 0; c < W; c++){
            if (c == 0)
                dp[r][c] = min(dp[r-1][c+1], dp[r-1][c]);
            else if (c == W-1)
                dp[r][c] = min(dp[r-1][c-1], dp[r-1][c]);
            else
                dp[r][c] = min({dp[r-1][c-1], dp[r-1][c], dp[r-1][c+1]});
            dp[r][c] += (int)image.at<uchar>(r, c);
        }
    }

    int min_value = 2147483647; // Infinity
    int min_index = -1;
    for(int c = 0; c < W; c++)
        if (dp[H-1][c] < min_value) {
            min_value = dp[H - 1][c];
            min_index = c;
        }

    int path[H];
    Point pos(H-1, min_index);
    path[pos.x] = pos.y;

    while (pos.x != 0){
        int value = dp[pos.x][pos.y] - (int)image.at<uchar>(pos.x, pos.y);
        int r = pos.x, c = pos.y;
        if (c == 0){
            if (value == dp[r-1][c+1])
                pos = Point(r-1, c+1);
            else
                pos = Point(r-1, c);
        }
        else if (c == W-1){
            if (value == dp[r-1][c-1])
                pos = Point(r-1, c-1);
            else
                pos = Point(r-1, c);
        }
        else{
            if (value == dp[r-1][c-1])
                pos = Point(r-1, c-1);
            else if (value == dp[r-1][c+1])
                pos = Point(r-1, c+1);
            else
                pos = Point(r-1, c);
        }
        path[pos.x] = pos.y;
    }

    return path;
}


void remove_pixels(Mat& image, Mat& output, int* seam){
    for(int r = 0; r < image.rows; r++ ) {
        for (int c = 0; c < image.cols; c++){
            if (c >= seam[r])
                output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c+1);
            else
                output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
        }
    }
}


void rot90(Mat &matImage, int rotflag){
    //1=CW, 2=CCW, 3=180
    if (rotflag == 1) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 1); // transpose+flip(1)=CW
    } else if (rotflag == 2) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 0); // transpose+flip(0)=CCW
    } else if (rotflag ==3) {
        flip(matImage, matImage, -1); // flip(-1)=180
    } else if (rotflag != 0) { // if not 0,1,2,3:
        cout  << "Unknown rotation flag(" << rotflag << ")" << endl;
    }
}


void remove_seam(Mat& image){
    int height = image.rows;
    int width = image.cols;

    // Make it gray scale first
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Run through energy function
    Mat emap;
    energy_function(gray, emap);
    int* seam = find_seam(emap);

    // Set up output matrix
    Mat output(height, width-1, CV_8UC3);
    remove_pixels(image, output, seam);
    image = output;
}


void shrink_image(Mat& image, int new_width, int width){
    cout << endl << "Processing image..." << endl;
    for(int i = 0; i < width - new_width; i++){
        remove_seam(image);
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./seamCarving <IMAGE_FILE>" << endl;
        return 1;
    }

    Mat image;
    image = imread(argv[1], 1);
    if (!image.data) {
        cout << "No image found" << endl;
        return -1;
    }

    cout << "The size of the image is: (" << image.cols << ", " << image.rows << ")" << endl;
    int new_width;
    cout << "Enter new width: ";
    cin >> new_width;
    shrink_image(image, new_width, image.cols);
    cout << "Done!" << endl;
    imwrite("output.jpg", image);
    return 0;
}

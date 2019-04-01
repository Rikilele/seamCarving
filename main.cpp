#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>


using namespace cv;
using namespace std;


void energy_function(Mat &image, Mat &output) {
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


int* find_seam(Mat &image, int min_col, int max_col) {
    int H = image.rows, W = image.cols;

    int dp[H][W];
    for(int c = 0; c < W; c++) {
        dp[0][c] = (int)image.at<uchar>(0, c);
    }

    for(int r = 1; r < H; r++) {
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


void remove_pixels(Mat& image, Mat& output, int* seam) {
    for (int r = 0; r < image.rows; r++ ) {
        for (int c = 0; c < image.cols; c++){
            if (c >= seam[r])
                output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c+1);
            else
                output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
        }
    }
}


/**
 * @param image - The image to remove a seam from
 * @param min - The minimum col of the previous iteration
 * @param max - The maximum col of the previous iteration
 * @return - Tuple of new min and max cols from this iteration
 */
tuple<int, int> remove_seam(Mat& image, int min, int max) {
    int height = image.rows;
    int width = image.cols;

    // Make it gray scale first
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Run through energy function
    Mat emap;
    energy_function(gray, emap);
    int* seam = find_seam(emap, min, max);

    // Set up output matrix
    Mat output(height, width-1, CV_8UC3);
    remove_pixels(image, output, seam);
    image = output;

    // Result
    int new_min = *min_element(seam, seam+height);
    int new_max = *max_element(seam, seam+height);
    return make_tuple(new_min, new_max);
}


void rotate_vid(VideoCapture vid, Mat* output, int width, int height, int depth) {
    for (int z = 0; z < depth; z++) {
        Mat frame;
        vid >> frame;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[width-x-1].at<Vec3b>(y, z) = frame.at<Vec3b>(y, x);
            }
        }
    }
}


void reverse_rotate_vid(Mat* input, Mat* output, int width, int height, int depth) {
    for (int z = 0; z < depth; z++) {
        Mat frame = input[z];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
//                cout << "x: " << x << " y: " << y << " z: " << depth-z-1 << endl;
                output[x].at<Vec3b>(y, depth-z-1) = frame.at<Vec3b>(y, x);
            }
        }
    }
}


int main(int argc, char** argv) {
    // Check command line arguments
    if (argc < 2) {
        cerr << "Usage: ./seamCarving <VIDEO_FILE>" << endl;
        return 1;
    }

    // Open the video file
    VideoCapture cap(argv[1]);

    // Check if media opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Calculate necessary info
    int fps = ceil(cap.get(CAP_PROP_FPS));
    int width = ceil(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = ceil(cap.get(CAP_PROP_FRAME_HEIGHT));
    int depth = ceil(cap.get(CAP_PROP_FRAME_COUNT));
    int target = depth * 2 / 3;

    cout << "W: " << width << " H: " << height << " D: " << depth << " T: " << target << endl;

    // Initialize new video data structure
    Mat* rotated_vid = new Mat[width];
    for (int i = 0; i < width; i++) {
        rotated_vid[i] = Mat(height, depth, CV_8UC3);
    }

    rotate_vid(cap, rotated_vid,  width, height, depth);

    /*
     * For each seam to be removed,
     * go through each frame to identify a seam
     * while incrementally restricting the band width
     */
    int min = 0;
    int max = depth;
    for (int left = depth; left > target; left--) {
        for (int i = 0; i < width; i++) {
            Mat frame = rotated_vid[i];
            tie(min, max) = remove_seam(frame, min, max);
        }
    }

    cout << "W: " << target << " H: " << height << " D: " << width << endl;

    // Initialize final video data structure
    Mat* final_vid = new Mat[target];
    for (int i = 0; i < target; i++) {
        final_vid[i] = Mat(height, width, CV_8UC3);
    }

    reverse_rotate_vid(rotated_vid, final_vid, target, height, width);

    // Produce output video as .avi
    VideoWriter video("out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
    for (int i = 0; i < target; i++) {
        Mat frame = final_vid[i];

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Write the frame into the file 'out.avi'
        video.write(frame);
    }

    cap.release();
    video.release();
    return 0;
}

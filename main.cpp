#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <ctime>

using namespace cv;
using namespace std;

//-------------------------------//
// Seam carving helper functions //
//-------------------------------//

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
    int tolerance = 0;
    int H = image.rows, W = image.cols;
    int left = (min_col > tolerance) ? min_col - tolerance : 0;
    int right = (max_col < W - tolerance) ? max_col + tolerance : W;

    // Get energy for first row
    int dp[H][max_col+tolerance*2];
    for(int c = left; c < right; c++) {
        dp[0][c] = (int)image.at<uchar>(0, c);
    }

    // Sift energy down
    for(int r = 1; r < H; r++) {
        for(int c = left; c < right; c++){
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
    for(int c = left; c < right; c++) {
        if (dp[H - 1][c] < min_value) {
            min_value = dp[H - 1][c];
            min_index = c;
        }
    }

    int path[H];
    Point pos(H-1, min_index);
    path[pos.x] = pos.y;

    while (pos.x != 0) {
        int value = dp[pos.x][pos.y] - (int)image.at<uchar>(pos.x, pos.y);
        int r = pos.x, c = pos.y;
        if (c == 0) {
            if (value == dp[r-1][c+1])
                pos = Point(r-1, c+1);
            else
                pos = Point(r-1, c);
        }
        else if (c == W-1) {
            if (value == dp[r-1][c-1])
                pos = Point(r-1, c-1);
            else
                pos = Point(r-1, c);
        }
        else {
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

/**
 * Seam-carves the {video} to fit the {target_depth}.
 *
 * @param video - The video to be seam-carved.
 * @param width - The width of the video.
 * @param depth - The depth (length in time) of the video.
 * @param target_depth - The new depth (length in time) of the video.
 */
void seam_carve_video(Mat* video, int width, int depth, int target_depth) {
    assert(depth >= target_depth);
    int min = 0;
    int max = depth;
    for (int left = depth; left > target_depth; left--) {
        for (int i = 0; i < width; i++) {
            tie(min, max) = remove_seam(video[i], min, max);
        }

        min = 0;
        max = depth;
    }
}

//--------------------------------------//
// Pre-post processing helper functions //
//--------------------------------------//

/**
 * Initializes and returns a prism matrix that can represent a video.
 *
 * @param w - The width of the prism.
 * @param h - The height of the prism.
 * @param d - The depth of the prism.
 * @return The prism matrix with specified dimensions.
 */
Mat* init_video_prism(int w, int h, int d) {
    Mat* cube = new Mat[d];
    for (int i = 0; i < d; i++) {
        cube[i] = Mat(h, w, CV_8UC3);
    }

    return cube;
}

/**
 * Read an {input} VideoCapture into a prism matrix {output},
 * but rotated about the height axis clockwise.
 *
 * @param input - The VideoCapture object to read the video from.
 * @param output - The rotated video represented as a prism matrix.
 * @param w - The width of the input video.
 * @param h - The height of the input video.
 * @param d - The depth of the input video.
 */
void rotate_vid(VideoCapture input, Mat* output, int w, int h, int d) {
    for (int z = 0; z < d; z++) {
        Mat frame;
        input >> frame;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                output[w-x-1].at<Vec3b>(y, z) = frame.at<Vec3b>(y, x);
            }
        }
    }
}

/**
 * Rotate the {input} prism matrix anti-clockwise on the height axis,
 * and store the result in the {output} prism matrix.
 *
 * @param input - The input prism matrix.
 * @param output - The output prism matrix.
 * @param w - The width of the input video.
 * @param h - The height of the input video.
 * @param d - The depth of the input video.
 */
void reverse_rotate_vid(Mat* input, Mat* output, int w, int h, int d) {
    for (int z = 0; z < d; z++) {
        Mat frame = input[z];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                output[x].at<Vec3b>(y, d-z-1) = frame.at<Vec3b>(y, x);
            }
        }
    }
}

/**
 * Compiles and exports the input prism matrix as an avi formatted video.
 *
 * @param vid - The video represented as a prism matrix.
 * @param fps - The fps of the exported video.
 * @param w - The width of the exported video.
 * @param h - The height of the exported video.
 * @param d - The depth of the exported video.
 * @param t - The title of the exported video.
 */
void export_video(Mat* vid, int fps, int w, int h, int d, const String& t) {
    VideoWriter out(
        t, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(w, h)
    );

    for (int i = 0; i < d; i++) {
        Mat frame = vid[i];
        if (frame.empty()) break;
        out.write(frame);
    }

    out.release();
}

//-----------------------------//
// Logistical helper functions //
//-----------------------------//

/**
 * Parses command line argument from main()
 * and returns corresponding VideoCapture object.
 * @param argc - Passed in from main().
 * @param argv - Passed in from main().
 * @return The VideoCapture object derived from the command line argument.
 */
VideoCapture parse_input(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./seamCarving <VIDEO_FILE>" << endl;
        exit(1);
    }

    VideoCapture cap(argv[1]);
    if(!cap.isOpened()) {
        cerr << "Error opening video stream or file" << endl;
        exit(-1);
    }

    return cap;
}

/**
 * Given a {start_time}, calculates and prints the split time.
 *
 * @param start_time - The start time of the split.
 * @param header - A message explaining the purpose of the split time.
 * @return The new start time of the next split.
 */
time_t take_split_time(time_t start_time, const String& header) {
    double duration = (clock() - start_time) / (double)CLOCKS_PER_SEC;
    cout << header << ": " << duration << endl;
    return clock();
}

//---------------//
// Main function //
//---------------//

/**
 * Increases playback speed of a video specified in the first command-line
 * argument by 1.5x, using a 3D seam-carving algorithm.
 *
 * Outputs the split time between major operations within the program.
 */
int main(int argc, char** argv) {

    // CP: Start of processing, open connection to video.
    clock_t start_time = clock();
    VideoCapture cap = parse_input(argc, argv);
    int fps = ceil(cap.get(CAP_PROP_FPS));
    int width = ceil(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = ceil(cap.get(CAP_PROP_FRAME_HEIGHT));
    int depth = ceil(cap.get(CAP_PROP_FRAME_COUNT));
    int target_depth = depth * 2 / 3;

    // CP: Finished opening connection to video, start rotating.
    start_time = take_split_time(start_time, "Opened video capture");
    Mat* rotated_vid = init_video_prism(depth, height, width);
    rotate_vid(cap, rotated_vid, width, height, depth);
    cap.release();

    // CP: Finished rotating video, start carving.
    start_time = take_split_time(start_time, "Rotated video");
    seam_carve_video(rotated_vid, width, depth, target_depth);

    // CP: Finished carving video, start re-rotating.
    start_time = take_split_time(start_time, "Carved video");
    Mat* final_vid = init_video_prism(width, height, target_depth);
    reverse_rotate_vid(rotated_vid, final_vid, target_depth, height, width);

    // CP: Finished re-rotating video, start exporting.
    start_time = take_split_time(start_time, "Re-rotated video");
    export_video(final_vid, fps, width, height, target_depth, "out.avi");

    // CP: Finished writing video to file.
    take_split_time(start_time, "Exported video");
    return 0;
}

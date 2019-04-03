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

void remove_pixels(Mat& image, Mat &output, int* seam) {
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
 * Given an energy map, finds a min-cut (a seam).
 *
 * @param energy_map - The energy map represented by Mat object.
 * @return An array of integers representing the min-cut.
 */
int* find_min_cut(Mat &energy_map){
    int height = energy_map.rows;
    int width = energy_map.cols;

    // Fill energy for first row.
    int dp[height][width];
    for (int c = 0; c < width; c++) {
        dp[0][c] = (int)energy_map.at<uchar>(0, c);
    }

    // Sift energy down.
    for (int r = 1; r < height; r++) {
        for(int c = 0; c < width; c++) {
            dp[r][c] += (int)energy_map.at<uchar>(r, c);

            // Add minimum energy from relevant cols above.
            if (c == 0) {
                dp[r][c] += min(dp[r-1][c+1], dp[r-1][c]);
            } else if (c == width - 1) {
                dp[r][c] += min(dp[r-1][c-1], dp[r-1][c]);
            } else {
                dp[r][c] += min({dp[r-1][c-1], dp[r-1][c], dp[r-1][c+1]});
            }
        }
    }

    // Find the minimum energy col on the bottom row.
    int min_val = 2147483647;
    int min_col = -1;
    for (int c = 0; c < width; c++) {
        if (dp[height-1][c] < min_val) {
            min_val = dp[height-1][c];
            min_col = c;
        }
    }

    // Store the seam in this array.
    int* path = new int[height];
    path[height-1] = min_col;

    // Keep appending the minimum energy from above.
    int col = min_col;
    for (int row = height-1; row > 0; row--) {

        // Left-most column.
        if (col == 0) {
            if (dp[row-1][col] < dp[row-1][col+1]) {
                path[row-1] = col;
            } else {
                path[row-1] = col + 1;
            }

        // Right-most column.
        } else if (col == width - 1) {
            if (dp[row-1][col] < dp[row-1][col-1]) {
                path[row-1] = col;
            } else {
                path[row-1] = col - 1;
            }

        // All other cases.
        } else {
            int diff = dp[row][col] - (int)energy_map.at<uchar>(row, col);

            // We prioritize the middle because it's in the same time frame.
            if (dp[row-1][col] == diff) {
                path[row-1] = col;
            } else if (dp[row-1][col-1] == diff) {
                path[row-1] = col - 1;
            } else {
                path[row-1] = col + 1;
            }
        }
    }

    return path;
}

/**
 * Given a frame of pixels, identifies and returns the min-energy seam.
 *
 * @param frame - A matrix reference of the frame.
 * @return An array of integers (which column to cut) representing a seam.
 */
int* find_seam_of_frame(Mat &frame) {

    //TODO: Might have to make it Mat& ^^^^

    // Make it gray scale first
    Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

    // Run through energy function
    Mat energy_map;
    energy_function(gray_frame, energy_map);

    return find_min_cut(energy_map);
}



/**
 * Seam-carves the input video to fit the target depth.
 *
 * @param vid - The video to be seam-carved.
 * @param w - The width of the video.
 * @param h - The height of the video.
 * @param d - The depth (length in time) of the video.
 * @param target_d - The new depth (length in time) of the video.
 */
void seam_carve_video(Mat* vid, int w, int h, int d, int target_d) {
    assert(d >= target_d);

    // Arrays to store seam information
    int* avg_seam = new int[h];
    int** all_seams = new int*[d];
    for(int i = 0; i < d; ++i) {
        all_seams[i] = new int[h];
    }

    // Repeat until video is shortened to target_d.
    for (int curr_d = d; curr_d > target_d; curr_d--) {

        // Go through each frame finding seams.
        for (int frame = 0; frame < w; frame++) {
            int* min_seam = find_seam_of_frame(vid[frame]);

            // Sum up the seams to calculate average later.
            for (int row = 0; row < h; row++) {
                avg_seam[row] += min_seam[row];
            }
        }

        // Find the average value of cols that has seams.
        for (int row = 0; row < h; row++) {
            avg_seam[row] = avg_seam[row] / w;
        }

        // Identify one frame with a seam closest to the average seam.
        int radix_frame = identify_closest_seam(all_seams, avg_seam);

        // Remove seams from frames after the radix
        int* prev_seam = all_seams[radix_frame];
        for (int frame = radix_frame; frame < w; frame++) {
            remove_seam(vid[frame], prev_seam);
        }

        // Remove seams from frames before the radix
        prev_seam = all_seams[radix_frame];
        for (int frame = radix_frame; frame > -1; frame--) {
            if (frame == radix_frame) continue;
            remove_seam(vid[frame], prev_seam);
        }
    }

    delete[] avg_seam;
    delete[] all_seams;
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
void export_video(Mat* vid, int fps, int w, int h, int d, const String &t) {
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
time_t take_split_time(time_t start_time, const String &header) {
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
    seam_carve_video(rotated_vid, height, width, depth, target_depth);

    // CP: Finished carving video, start re-rotating.
    start_time = take_split_time(start_time, "Carved video");
    Mat* final_vid = init_video_prism(width, height, target_depth);
    reverse_rotate_vid(rotated_vid, final_vid, target_depth, height, width);
    delete[] rotated_vid;

    // CP: Finished re-rotating video, start exporting.
    start_time = take_split_time(start_time, "Re-rotated video");
    export_video(final_vid, fps, width, height, target_depth, "out.avi");
    delete[] final_vid;

    // CP: Finished writing video to file.
    take_split_time(start_time, "Exported video");
    return 0;
}

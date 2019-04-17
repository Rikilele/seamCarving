#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;

//-------------------------------//
// Seam carving helper functions //
//-------------------------------//

/**
 * Maps an input {frame} in gray scale to an {output} via an energy map.
 * Directly sourced from {@link https://github.com/loc-trinh/seamCarving}.
 *
 * @param frame - The gray scale frame.
 * @param output - The energy map of the frame.
 */
void energy_function(Mat &frame, Mat &output) {
    Mat dx, dy;
    Sobel(frame, dx, CV_64F, 1, 0);
    Sobel(frame, dy, CV_64F, 0, 1);
    magnitude(dx, dy, output);

    double min_value, max_value, z;
    minMaxLoc(output, &min_value, &max_value);
    z = 1 / max_value * 255;
    output = output * z; // Normalize
    output.convertTo(output, CV_8U);
}

/**
 * Given an energy map, finds a min-cut (a seam).
 *
 * @param energy_map - The energy map represented by Mat object.
 * @return An array of integers representing the min-cut.
 */
int* find_min_cut(Mat &energy_map) {
    int height = energy_map.rows;
    int width = energy_map.cols;

    // Fill energy for first row.
    int dp[height][width];
    for (int c = 0; c < width; c++) {
        dp[0][c] = (int)energy_map.at<uchar>(0, c);
    }

    // Sift energy down.
    for (int r = 1; r < height; r++) {
        for (int c = 0; c < width; c++) {
            dp[r][c] = (int)energy_map.at<uchar>(r, c);

            // Add minimum energy from relevant cols above.
            if (c == 0) {
                dp[r][c] = dp[r][c] + min(dp[r-1][c+1], dp[r-1][c]);
            } else if (c == width - 1) {
                dp[r][c] = dp[r][c] + min(dp[r-1][c-1], dp[r-1][c]);
            } else {
                dp[r][c] = dp[r][c] + min({dp[r-1][c-1], dp[r-1][c], dp[r-1][c+1]});
            }
        }
    }

    // Find the minimum energy col on the bottom row.
    int min_val = INT_MAX;
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
            if (dp[row-1][col] > dp[row-1][col+1]) {
                col = col + 1;
            }

        // Right-most column.
        } else if (col == width - 1) {
            if (dp[row-1][col] > dp[row-1][col-1]) {
                col = col - 1;
            }

        // All other cases.
        } else {
            int diff = dp[row][col] - (int)energy_map.at<uchar>(row, col);

            // We prioritize the middle because it's in the same time frame.
            if (dp[row-1][col] == diff) {
                col = col;
            } else if (dp[row-1][col-1] == diff) {
                col = col - 1;
            } else {
                col = col + 1;
            }
        }

        path[row-1] = col;
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
    Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
    Mat energy_map;
    energy_function(gray_frame, energy_map);
    return find_min_cut(energy_map);
}

/**
 * Removes a {seam} from a {frame}.
 *
 * @param frame - The frame to remove the seam from.
 * @param seam - The seam to remove from the frame.
 * @return The Mat holding the new frame with the removed seam.
 */
Mat remove_seam(Mat &frame, const int* seam) {
    int height = frame.rows;
    int width = frame.cols;

    Mat output(height, width-1, CV_8UC3);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            if (c >= seam[r]) {
                output.at<Vec3b>(r, c) = frame.at<Vec3b>(r, c + 1);
            } else {
                output.at<Vec3b>(r, c) = frame.at<Vec3b>(r, c);
            }
        }
    }

    return output;
}

//----------------------------------//
// Seam comparison helper functions //
//----------------------------------//

/**
 * Calculates the difference between two seams.
 *
 * @param a - The first seam.
 * @param b - The second seam.
 * @param size - The size (height) of the seam.
 * @return The difference (magnitude) of the seam.
 */
float calc_seam_diff(int* a, int* b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum = sum + pow(a[i] - b[i], 2);
    }

    return sqrt(sum);
}

/**
 * Finds the index of the seam inside {seams} closest to {ref}.
 *
 * @param ref - The reference seam.
 * @param seam_size - The size of the seam.
 * @param seams - An array of seams.
 * @param arr_size - The size of the array of seams.
 * @return The index of the closest seam.
 */
int identify_closest_seam(int* ref, int seam_size, int** arr, int arr_size) {
    int best_seam = 0;
    float min_delta = calc_seam_diff(arr[0], ref, seam_size);
    for (int i = 1; i < arr_size; i++) {
        float delta = calc_seam_diff(arr[i], ref, seam_size);
        if (delta < min_delta) {
            best_seam = i;
            min_delta = delta;
        }
    }

    return best_seam;
}

//-------------------------------------------//
// Incremental seam-carving helper functions //
//-------------------------------------------//

int* find_incremental_seam(Mat &frame, const int* prev_seam) {

    // Create an energy map.
    Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
    Mat energy_map;
    energy_function(gray_frame, energy_map);

    // Fill all cells other than +/- 1 pixels around prev_seam with infinity.
    for (int r = 0; r < energy_map.rows; r++) {

        // Find the "no-touch zone"
        int center_col = prev_seam[r];
        int left_bound = max(center_col-1, 0);
        int right_bound = min(center_col+1, energy_map.cols);

        // Cross out columns before "no-touch"
        for (int c = 0; c < left_bound; c++) {
            energy_map.at<uchar>(r, c) = (uchar)INT_MAX;
        }

        // Cross out columns after "no-touch"
        for (int c = right_bound + 1; c < energy_map.cols; c++) {
            energy_map.at<uchar>(r, c) = (uchar)INT_MAX;
        }
    }

    return find_min_cut(energy_map);
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
// Main seam-carving algorithm //
//-----------------------------//

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

    // Init arrays to store seam information
    int* avg_seam = new int[h];
    int** all_seams = new int*[w];
    for (int i = 0; i < w; i++) {
        all_seams[i] = new int[h];
    }

    // Repeat until video is shortened to target_d.
    for (int curr_d = d; curr_d > target_d; curr_d--) {

        // Reset avg_seam
        for (int i = 0; i < h; i++) {
            avg_seam[i] = 0;
        }

        cout << "------- LEFT: " << curr_d - target_d << " ---------" << endl;

        // Go through each frame finding seams.
        for (int frame = 0; frame < w; frame++) {
            int* min_seam = find_seam_of_frame(vid[frame]);
            all_seams[frame] = min_seam;

            // Sum up the seams to calculate average later.
            for (int row = 0; row < h; row++) {
                avg_seam[row] = avg_seam[row] + min_seam[row];
            }
        }

        // Find the average value of cols that has seams.
        for (int row = 0; row < h; row++) {
            avg_seam[row] = avg_seam[row] / w;
        }

        // Identify one frame with a seam closest to the average seam.
        int radix_frame = identify_closest_seam(avg_seam, h, all_seams, w);

        // Remove the seam from the radix.
        int* cut_seam = all_seams[radix_frame];

        vid[radix_frame] = remove_seam(vid[radix_frame], cut_seam);

        // Remove seams from frames before the radix
        Mat frame;
        for (int i = radix_frame - 1; i >= 0; i--) {
            cut_seam = find_incremental_seam(vid[i], cut_seam);
            vid[i] = remove_seam(vid[i], cut_seam);
        }

        // Remove seams from frames after the radix
        cut_seam = all_seams[radix_frame];
        for (int i = radix_frame + 1; i < w; i++) {
            cut_seam = find_incremental_seam(vid[i], cut_seam);
            vid[i] = remove_seam(vid[i], cut_seam);
        }
    }

    delete[] avg_seam;
    delete[] all_seams;
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

    cout << "w:" << width << " h:" << height << " d:" << depth << endl;

    // CP: Finished opening connection to video, start rotating.
    start_time = take_split_time(start_time, "Opened video capture");
    Mat* rotated_vid = init_video_prism(depth, height, width);
    rotate_vid(cap, rotated_vid, width, height, depth);
    cap.release();

    // CP: Finished rotating video, start carving.
    start_time = take_split_time(start_time, "Rotated video");
    seam_carve_video(rotated_vid, width, height, depth, target_depth);

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

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;
namespace fs = std::filesystem;  // For directory iteration

// Define a 3x3 Vertical Edge Detection Kernel
const int KERNEL_SIZE = 3;
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    { 1,  0, -1 },
    { 1,  0, -1 },
    { 1,  0, -1 }
};

// Function to perform manual 2D convolution (Scalar)
Mat applyConvolutionScalar(const Mat& input) {
    int rows = input.rows;
    int cols = input.cols;
    Mat output = Mat::zeros(rows, cols, CV_32F); // Floating-point output

    int pad = KERNEL_SIZE / 2; // Padding size

    // Convert input to float to avoid data loss
    Mat floatInput;
    input.convertTo(floatInput, CV_32F);

    // Pad the input image using BORDER_REPLICATE (better for edges)
    Mat paddedInput;
    copyMakeBorder(floatInput, paddedInput, pad, pad, pad, pad, BORDER_REPLICATE);

    // Iterate over each pixel (excluding borders)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0.0;

            // Apply kernel
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    int x = i + ki;
                    int y = j + kj;
                    sum += paddedInput.at<float>(x, y) * kernel[ki][kj];
                }
            }

            output.at<float>(i, j) = sum;
        }
    }

    // Convert the output to absolute values to enhance edges
    output = abs(output);

    // Normalize and convert back to 8-bit grayscale image
    Mat normalizedOutput;
    normalize(output, normalizedOutput, 0, 255, NORM_MINMAX);
    normalizedOutput.convertTo(normalizedOutput, CV_8U);

    return normalizedOutput;
}

int main() {
    string inputFolder = "/home/rehan/Assignments/PDC/i220965_A_A3/DataSet";  // Folder containing input images
    string outputFolder = "/home/rehan/Assignments/PDC/i220965_A_A3/OutputQ1/"; // Folder to save processed images

    // Ensure output folder exists
    fs::create_directories(outputFolder);

    int count = 0;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file()) {
            string inputPath = entry.path().string();
            string fileName = entry.path().filename().string();
            string outputPath = outputFolder + "processed_" + fileName;

            // Load the image in grayscale
            Mat image = imread(inputPath, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "Error: Could not load image " << inputPath << endl;
                continue;
            }

            cout << "Processing: " << fileName << " (" << image.rows << "x" << image.cols << ")" << endl;

            // Measure execution time
            auto start = high_resolution_clock::now();
            Mat edgeDetectedImage = applyConvolutionScalar(image);
            auto end = high_resolution_clock::now();

            duration<double> elapsed = end - start;
            cout << "Execution time: " << fixed << elapsed.count() << " seconds" << endl;

            // Save the processed image
            imwrite(outputPath, edgeDetectedImage);
            cout << "Saved: " << outputPath << endl;

            count++;
        }
    }

    cout << "\nProcessing complete! Total images processed: " << count << endl;
    return 0;
}


// Rehan Tariq
// 22i-0965
// CS-6A

#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace std::chrono;

// Function to read OpenCL kernel file
string readKernelFile(const string& fileName) {
    ifstream file(fileName);
    if (!file.is_open()) {
        cerr << "Error: Could not open OpenCL kernel file!" << endl;
        exit(1);
    }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

int main() {
    string inputFolder = "/home/rehan/Assignments/PDC/i220965_A_A3/DataSet";
    string outputFolder = "./processed/";
    fs::create_directory(outputFolder);

    // Read OpenCL kernel source
    string kernelSource = readKernelFile("convolution1.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    // Select OpenCL platform and device
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program program(context, sources);

    if (program.build({device}) != CL_SUCCESS) {
        cerr << "Error: Failed to build OpenCL program!" << endl;
        cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        return -1;
    }

    // Define edge detection kernel (Sobel)
    vector<float> kernel = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    int kernelSize = 3;

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file()) {
            string inputPath = entry.path().string();
            string outputPath = outputFolder + entry.path().stem().string() + "_processed.jpg";
            
            // Load image in grayscale
            Mat image = imread(inputPath, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "Error: Could not load image " << inputPath << endl;
                continue;
            }

            int width = image.cols;
            int height = image.rows;
            int imageSize = width * height;
            
            // Convert image to float vector
            vector<float> inputImage(imageSize);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    inputImage[i * width + j] = static_cast<float>(image.at<uchar>(i, j));
                }
            }

            vector<float> outputImage(imageSize, 0);
            
            // Create OpenCL buffers
            cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * imageSize, inputImage.data());
            cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * imageSize);
            cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kernel.size(), kernel.data());
            
            // Set kernel arguments
            cl::Kernel convKernel(program, "convolution");
            convKernel.setArg(0, inputBuffer);
            convKernel.setArg(1, outputBuffer);
            convKernel.setArg(2, kernelBuffer);
            convKernel.setArg(3, width);
            convKernel.setArg(4, height);
            convKernel.setArg(5, kernelSize);

            // Execute kernel
            cl::NDRange globalSize(width, height);
            auto start = high_resolution_clock::now();
            queue.enqueueNDRangeKernel(convKernel, cl::NullRange, globalSize, cl::NullRange);
            queue.finish();
            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;
            cout << "Processed " << inputPath << " in " << elapsed.count() << " seconds." << endl;

            // Read output buffer
            queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * imageSize, outputImage.data());

            // Normalize output to 0-255 range
            float maxVal = *max_element(outputImage.begin(), outputImage.end());
            float minVal = *min_element(outputImage.begin(), outputImage.end());
            if (maxVal > minVal) {
                for (int i = 0; i < imageSize; i++) {
                    outputImage[i] = ((outputImage[i] - minVal) / (maxVal - minVal)) * 255.0f;
                }
            }

            // Convert to OpenCV image
            Mat result(height, width, CV_8U);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    result.at<uchar>(i, j) = static_cast<uchar>(outputImage[i * width + j]);
                }
            }

            // Save output image
            imwrite(outputPath, result);
            cout << "Saved: " << outputPath << endl;
        }
    }

    return 0;
}


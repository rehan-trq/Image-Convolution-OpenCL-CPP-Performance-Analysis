// Rehan Tariq
// 22i-0965
// CS-6A

__kernel void convolution(
    __global const float* input,
    __global float* output,
    __constant float* filter,
    int width,
    int height,
    int filterSize
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfSize = filterSize / 2;

    if (x >= width || y >= height) return;

    float4 sum = (float4)(0.0f);

    for (int ky = -halfSize; ky <= halfSize; ky++) {
        for (int kx = -halfSize; kx <= halfSize; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int imgIndex = iy * width + ix;
                int filterIndex = (ky + halfSize) * filterSize + (kx + halfSize);
                
                // Load 4 pixels at once using vectorized data type
                float4 pixel = vload4(0, &input[imgIndex]);
                float4 kernelVal = (float4)(filter[filterIndex]);
                sum += pixel * kernelVal;
            }
        }
    }

    // Reduce sum from float4 to scalar
    float finalSum = sum.x + sum.y + sum.z + sum.w;
    finalSum = fabs(finalSum);
    output[y * width + x] = finalSum;
} 


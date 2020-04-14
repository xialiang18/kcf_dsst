#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "fhogcuda.h"

#define THREAD_BUNDLE_NUM 128

__constant__ float boundary_x[NUM_SECTOR + 1];
__constant__ float boundary_y[NUM_SECTOR + 1];

#define MAX_CELL_SIZE 16
__constant__ int nearest[MAX_CELL_SIZE];
__constant__ float w[MAX_CELL_SIZE * 2];

/*__device__ inline int mmin(int a, int b){
    return a > b ? b : a;
}*/

#define min(a, b) (a < b ? a : b)

__global__ void d_imageGrad(unsigned char *imageData, float *dx, float *dy, cv::Size size)
{
    __shared__ unsigned char image[4][32];
    const unsigned int idx = threadIdx.x;
    const unsigned int idy = threadIdx.y;
    const unsigned int x = (blockIdx.x * 30) + threadIdx.x;
    const unsigned int y = (blockIdx.y * 2) + threadIdx.y;

    int target_index = y * size.width + x;
    image[idy][idx] = (unsigned char)imageData[target_index];
    __syncthreads();

    if(idx > 0 && idx < 31 && idy > 0 && idy < 3 && x < size.width - 1 && y < size.height - 1){
        int x1 = image[idy][idx - 1];
        int x2 = image[idy][idx + 1];
        int y1 = image[idy - 1][idx];
        int y2 = image[idy + 1][idx];
        dx[target_index] = x2 - x1;
        dy[target_index] = y2 - y1;
    }
}

__global__ void d_maxGrad(float *dx, float *dy, float *r, int *alfa, cv::Size size, int channels)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(idx > 0 && idx < size.width - 1 && idy > 0 && idy < size.height - 1){
        //int c = 0;
        int pos = idy * size.width * channels + idx * channels;
        float x, y;
        float tx, ty;
        float magnitude;
        float max_r = 0;
        //r[idy * size.width + idx] = sqrtf(x * x + y * y);
        for(int ch = 0; ch < channels; ch++)
        {
            tx = (dx[pos + ch]);
            ty = (dy[pos + ch]);
            magnitude = sqrtf(tx * tx + ty * ty);
            if(magnitude > max_r)
            {
                max_r = magnitude;
                //c = ch;
                x = tx;
                y = ty;
            }
        }

        r[idy * size.width + idx] = max_r;

        float max  = boundary_x[0] * x + boundary_y[0] * y;
        float maxi = 0;
        float dotProd;
        for (int kk = 0; kk < NUM_SECTOR; kk++)
        {
            dotProd = boundary_x[kk] * x + boundary_y[kk] * y;
            if (dotProd > max)
            {
                max  = dotProd;
                maxi = kk;
            }
            else
            {
                if (-dotProd > max)
                {
                    max  = -dotProd;
                    maxi = kk + NUM_SECTOR;
                }
            }
        }
        alfa[idy * size.width * 2 + idx * 2    ] = maxi > NUM_SECTOR ? maxi - NUM_SECTOR : maxi;
        alfa[idy * size.width * 2 + idx * 2 + 1] = maxi;
    }
}

__global__ void d_featureMaps(float *r, int *alfa, float *map, int k, int sizeX, int sizeY, int p, cv::Size size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int i = idy / k;
    int j = idx / k;
    int ii = idy - i * k;
    int jj = idx - j * k;
    int width = size.width;
    int height = size.height;

    if(j < width && i < height){
        // int ii = t / k;
        // int jj = t - ii * k;
        // int width = size.width;
        // int height = size.height;
        int stringSize = sizeX * p;

        if ((i * k + ii > 0) && (i * k + ii < height - 1) && (j * k + jj > 0) && (j * k + jj < width  - 1))
        {
          int d = (k * i + ii) * width + (j * k + jj);
          int alfa_2d = alfa[d * 2];
          int alfa_2d_1 = alfa[d * 2 + 1];
          float rr = r[d];
          float increment = rr * w[ii * 2] * w[jj * 2];
          atomicAdd(&map[ i * stringSize + j * p + alfa_2d], increment);
          atomicAdd(&map[ i * stringSize + j * p + alfa_2d_1 + NUM_SECTOR], increment);
          if ((i + nearest[ii] >= 0) &&
              (i + nearest[ii] <= sizeY - 1))
          {
            increment = rr * w[ii * 2 + 1] * w[jj * 2];
            atomicAdd(&map[(i + nearest[ii]) * stringSize + j * p + alfa_2d], increment);
            atomicAdd(&map[(i + nearest[ii]) * stringSize + j * p + alfa_2d_1 + NUM_SECTOR], increment);
          }
          if ((j + nearest[jj] >= 0) &&
              (j + nearest[jj] <= sizeX - 1))
          {
            increment = rr * w[ii * 2] * w[jj * 2 + 1];
            atomicAdd(&map[i * stringSize + (j + nearest[jj]) * p + alfa_2d], increment);
            atomicAdd(&map[i * stringSize + (j + nearest[jj]) * p + alfa_2d_1 + NUM_SECTOR], increment);
          }
          if ((i + nearest[ii] >= 0) &&
              (i + nearest[ii] <= sizeY - 1) &&
              (j + nearest[jj] >= 0) &&
              (j + nearest[jj] <= sizeX - 1))
          {
            increment = rr * w[ii * 2 + 1] * w[jj * 2 + 1];
            atomicAdd(&map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa_2d], increment);
            atomicAdd(&map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa_2d_1 + NUM_SECTOR], increment);
          }
        }
    }
}

__global__ void d_squareSum(float* map, float* partOfNorm, int numFeatures, int num, int size)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size){
        float valOfNorm = 0.0f;
        int pos = id * numFeatures;
        for(int j = 0; j < num; j++)
        {
            valOfNorm += map[pos + j] * map[pos + j];
        }
        partOfNorm[id] = valOfNorm;
    }
}

__global__ void d_normalization(float *map, float* partOfNorm, float * newData, cv::Size size, float alfa)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int target_x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int target_y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    __shared__ float norm[4][32];
    int width = size.width;
    int height = size.height;
    if(idx < size.width && idy < size.height){
        norm[y][x] = partOfNorm[target_y * width + target_x];
    }

    if(idx >= 1 && idx < size.width - 1 && idy >= 1 && idy < size.height - 1 && x >= 1 && x <= 30 && y >= 1 && y <= 2){
        int xp = 3 * NUM_SECTOR;
        int pp = 12 * NUM_SECTOR;
        int sizeX = size.width - 2;
        int pos1 = (idy  ) * (sizeX + 2) * xp + (idx  ) * xp;
        int pos2 = (idy - 1) * (sizeX  ) * pp + (idx - 1) * pp;
        float valOfNorm1 = sqrt(
            norm[y][x] +
            norm[y][x + 1] +
            norm[y + 1][x] +
            norm[y + 1][x + 1]) + FLT_EPSILON;

        float valOfNorm2 = sqrt(
            norm[y][x] +
            norm[y][x + 1] +
            norm[y - 1][x] +
            norm[y - 1][x + 1]) + FLT_EPSILON;

        float valOfNorm3 = sqrt(
            norm[y][x] +
            norm[y][x - 1] +
            norm[y + 1][x] +
            norm[y + 1][x - 1]) + FLT_EPSILON;

        float valOfNorm4 = sqrt(
            norm[y][x] +
            norm[y][x - 1] +
            norm[y - 1][x] +
            norm[y - 1][x - 1]) + FLT_EPSILON;

        float map_idz = map[pos1 + idz];
        float map_idz_2 = map[pos1 + 2 * idz + NUM_SECTOR];
        float map_idz_2_1 = map[pos1 + 2 * idz + NUM_SECTOR + 1];

        newData[pos2 + idz] = min(map_idz / valOfNorm1, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 4] = min(map_idz_2 / valOfNorm1, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 4 + 1] = min(map_idz_2_1 / valOfNorm1, alfa);

        newData[pos2 + idz + NUM_SECTOR] = min(map_idz / valOfNorm2, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 6] = min(map_idz_2 / valOfNorm2, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 6 + 1] = min(map_idz_2_1 / valOfNorm2, alfa);

        newData[pos2 + idz + NUM_SECTOR * 2] = min(map_idz / valOfNorm3, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 8] = min(map_idz_2 / valOfNorm3, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 8 + 1] = min(map_idz_2_1 / valOfNorm3, alfa);

        newData[pos2 + idz + NUM_SECTOR * 3] = min(map_idz / valOfNorm4, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 10] = min(map_idz_2 / valOfNorm4, alfa);
        newData[pos2 + 2 * idz + NUM_SECTOR * 10 + 1] = min(map_idz_2_1 / valOfNorm4, alfa);
    }
}

__global__ void d_PCAMaps(float *newData, float *featureData, cv::Size size, int xp, int yp)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float nx    = 1.0f / sqrtf((float)(xp * 2));
    float ny    = 1.0f / sqrtf((float)(yp    ));
    int val = 0;

    if(idx < size.width * size.height){
        int pos1 = idx * NUM_SECTOR * 12;
        int pos2 = idx * (NUM_SECTOR * 3 + 4);
        int k = 0;
        int ii, jj;
        for(jj = 0; jj < xp * 2; jj++)
        {
            val = 0;
            for(ii = 0; ii < yp; ii++)
            {
                val += newData[pos1 + yp * xp + ii * xp * 2 + jj];
            }
            featureData[pos2 + k] = val * ny;
            k++;
        }
        for(jj = 0; jj < xp; jj++)
        {
            val = 0;
            for(ii = 0; ii < yp; ii++)
            {
                val += newData[pos1 + ii * xp + jj];
            }
            featureData[pos2 + k] = val * ny;
            k++;
        }
        for(ii = 0; ii < yp; ii++)
        {
            val = 0;
            for(jj = 0; jj < 2 * xp; jj++)
            {
                val += newData[pos1 + yp * xp + ii * xp * 2 + jj];
            }
            featureData[pos2 + k] = val * nx;
            k++;
        }
    }
}

void fhogCudaInit(int cell_size)
{
    if(cell_size > MAX_CELL_SIZE){
        std::cout << "MAX_CELL_SIZE is too small, please make it larger!" << std::endl;
    }
    float h_boundary_x[NUM_SECTOR + 1];
    float h_boundary_y[NUM_SECTOR + 1];

    float arg_vector;
    for(int i = 0; i <= NUM_SECTOR; i++)
    {
        arg_vector    = ( (float) i ) * ( (float)(PI) / (float)(NUM_SECTOR) );
        h_boundary_x[i] = cosf(arg_vector);
        h_boundary_y[i] = sinf(arg_vector);
    }

    cudaMemcpyToSymbol(boundary_x, h_boundary_x, sizeof(float) * (NUM_SECTOR + 1));
    cudaMemcpyToSymbol(boundary_y, h_boundary_y, sizeof(float) * (NUM_SECTOR + 1));

    int k = cell_size;
    int h_nearest[k];
    float h_w[2 * k];

    for(int i = 0; i < k / 2; i++)
    {
        h_nearest[i] = -1;
    }/*for(i = 0; i < k / 2; i++)*/
    for(int i = k / 2; i < k; i++)
    {
        h_nearest[i] = 1;
    }/*for(i = k / 2; i < k; i++)*/

    float a_x, b_x;
    for(int j = 0; j < k / 2; j++)
    {
        b_x = k / 2 + j + 0.5f;
        a_x = k / 2 - j - 0.5f;
        h_w[j * 2    ] = 1.0f / a_x * ((a_x * b_x) / ( a_x + b_x));
        h_w[j * 2 + 1] = 1.0f / b_x * ((a_x * b_x) / ( a_x + b_x));
    }/*for(j = 0; j < k / 2; j++)*/
    for(int j = k / 2; j < k; j++)
    {
        a_x = j - k / 2 + 0.5f;
        b_x =-j + k / 2 - 0.5f + k;
        h_w[j * 2    ] = 1.0f / a_x * ((a_x * b_x) / ( a_x + b_x));
        h_w[j * 2 + 1] = 1.0f / b_x * ((a_x * b_x) / ( a_x + b_x));
    }/*for(j = k / 2; j < k; j++)*/

    cudaMemcpyToSymbol(w, h_w, sizeof(float) * (k * 2));
    cudaMemcpyToSymbol(nearest, h_nearest, sizeof(int  ) *  k);
}

void imageGrad(unsigned char *imageData, float *dxData, float *dyData, cv::Size size)
{
    //dim3 threads(THREAD_BUNDLE_NUM, 1);
    dim3 threads(32, 4);
    int blockNum_x = (size.width + 30 - 1) / 30;
    int blockNum_y = (size.height + 4 - 1) / 4;
    dim3 blocks(blockNum_x, blockNum_y);

    d_imageGrad<<<blocks, threads>>>(imageData, dxData, dyData, size);
}

void maxGrad(float *dx, float *dy, float *r, int *alfa, cv::Size size, int channels)
{
    dim3 threads(THREAD_BUNDLE_NUM, 1);
    int numX = (size.width + THREAD_BUNDLE_NUM - 1) / THREAD_BUNDLE_NUM;
    int numY = size.height;
    dim3 blocks(numX, numY);

    d_maxGrad<<<blocks, threads>>>(dx, dy, r, alfa, size, channels);
}

void featureMaps(float *r, int *alfa, float *map, int k, int sizeX, int sizeY, int p, cv::Size size)
{
    // dim3 threads(32, 1, 1);
    // dim3 blocks((sizeX + 31) / 32, sizeY, k * k);
    int width = size.width;
    int height = size.height;

    int thread_x, thread_y, block_x, block_y;
    if(width > 128){
        thread_x = 128;
        thread_y = 1;
        block_x = (width + 127) / 128;
        block_y = height;
    }else{
        //thread_x = width;
        thread_y = 128 / width;
        thread_x = 128 / thread_y;
        block_x = 1;
        block_y = (height + thread_y - 1) / thread_y;
    }

    dim3 threads(thread_x, thread_y);
    dim3 blocks(block_x, block_y);

    d_featureMaps<<<blocks, threads>>>(r, alfa, map, k, sizeX, sizeY, p, size);
}

void squareSum(float* map, float* partOfNorm, int numFeatures, int num, int size)
{
    int thread = THREAD_BUNDLE_NUM;
    int block = (size + THREAD_BUNDLE_NUM - 1) / THREAD_BUNDLE_NUM;
    d_squareSum<<<block, thread>>>(map, partOfNorm, numFeatures, num, size);
}


void normalization(float *map, float* partOfNorm, float * newData, cv::Size size, float alfa)
{
    // dim3 threads(32, 1, 1);
    // dim3 blocks((size.width + 31) / 32, size.height, NUM_SECTOR);
    int width = size.width + 2;
    int height = size.height + 2;
    int thread_x, thread_y, block_x, block_y;

    if(width > 128){
        thread_x = 128;
        thread_y = 3;
        block_x = (width + 125) / 126;
        block_y = height;
    }else{
        thread_y = 128 / width;
        thread_x = 128 / thread_y;
        block_x = 1;
        block_y = (height + thread_y - 1) / thread_y;
    }

    thread_x = 32;
    thread_y = 4;
    block_x = (width + 30 - 1) / 30;
    block_y = (height + 2 - 1) / 2;
    dim3 threads(32, 4, 1);
    dim3 blocks(block_x, block_y, NUM_SECTOR);

    d_normalization<<<blocks, threads>>>(map, partOfNorm, newData, size, alfa);
}

void PCAMaps(float *newData, float *featureData, cv::Size size, int xp, int yp)
{
    //dim3 threads(32, 1);
    //dim3 blocks((size.width + 31) / 32, size.height);
    int threads = THREAD_BUNDLE_NUM;
    int blocks = (size.height * size.width + THREAD_BUNDLE_NUM - 1) / THREAD_BUNDLE_NUM;

    d_PCAMaps<<<blocks, threads>>>(newData, featureData, size, xp, yp);
}

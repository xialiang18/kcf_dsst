#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "kcftracker.hpp"

//const int nClusters = 15;
__constant__ float data[15][3] = {
{161.317504, 127.223401, 128.609333},
{142.922425, 128.666965, 127.532319},
{67.879757, 127.721830, 135.903311},
{92.705062, 129.965717, 137.399500},
{120.172257, 128.279647, 127.036493},
{195.470568, 127.857070, 129.345415},
{41.257102, 130.059468, 132.675336},
{12.014861, 129.480555, 127.064714},
{226.567086, 127.567831, 136.345727},
{154.664210, 131.676606, 156.481669},
{121.180447, 137.020793, 153.433743},
{87.042204, 137.211742, 98.614874},
{113.809537, 106.577104, 157.818094},
{81.083293, 170.051905, 148.904079},
{45.015485, 138.543124, 102.402528}};

float data1[15][3] = {
    {161.317504, 127.223401, 128.609333},
    {142.922425, 128.666965, 127.532319},
    {67.879757, 127.721830, 135.903311},
    {92.705062, 129.965717, 137.399500},
    {120.172257, 128.279647, 127.036493},
    {195.470568, 127.857070, 129.345415},
    {41.257102, 130.059468, 132.675336},
    {12.014861, 129.480555, 127.064714},
    {226.567086, 127.567831, 136.345727},
    {154.664210, 131.676606, 156.481669},
    {121.180447, 137.020793, 153.433743},
    {87.042204, 137.211742, 98.614874},
    {113.809537, 106.577104, 157.818094},
    {81.083293, 170.051905, 148.904079},
    {45.015485, 138.543124, 102.402528}};

void test(cv::Size z, int cell_size, unsigned char *input, float *out, cv::Size outsize);

__global__ void d_getLabFeatures(cv::Size z, int cell_size, unsigned char *input, float *out, cv::Size outsize)
{
    // if (threadIdx.x == 0) 
    //     printf("Hello thread %d, f=%f\n", threadIdx.x, 0.55) ;
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    // printf("hello world!");
    // if(idx < z.width && idy < z.height)
    //     cnt[z.width * idy + idx] = 10;
    if(idx >= cell_size && idx < z.width - cell_size && idy >= cell_size && idy < z.height - cell_size){
        int cntCell = idx / cell_size - 1 + (idy / cell_size - 1) * ( z.width / cell_size - 2);
        //cnt[z.width * idy + idx] = 10;
        float l = (float)input[(z.width * idy + idx) * 3];
        float a = (float)input[(z.width * idy + idx) * 3 + 1];
        float b = (float)input[(z.width * idy + idx) * 3 + 2];

        float minDist = FLT_MAX;
        int minIdx = 0;
        for(int k = 0; k < outsize.height; ++k){
            float dist = ( (l - data[k][0]) * (l - data[k][0]) )
                       + ( (a - data[k][1]) * (a - data[k][1]) ) 
                       + ( (b - data[k][2]) * (b - data[k][2]) );
            if(dist < minDist){
                minDist = dist;
                minIdx = k;
            }
        }

        //out[minIdx * outsize.width + cntCell] += 1.0 / (cell_size * cell_size);
        atomicAdd(out + minIdx * outsize.width + cntCell, 1.0 / (cell_size * cell_size));
    }
}

void getLabFeatures(cv::Size z, int cell_size, unsigned char *input, float *out, cv::Size outsize)
{
    int width = z.width;
    int height = z.height;
    //dim3 thread, block;
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

    dim3 thread(thread_x, thread_y);
    dim3 block(block_x, block_y);

    cudaMemset(out, 0, sizeof (float) * (outsize.width * outsize.height));
    d_getLabFeatures<<<block, thread>>>(z, cell_size, input, out, outsize);
    cudaDeviceSynchronize();
}

void test(cv::Size z, int cell_size, unsigned char *input, float *out, cv::Size outsize)
{
    int cntCell = 0;
    for (int cY = cell_size; cY < z.height - cell_size; cY += cell_size){
        for (int cX = cell_size; cX < z.width - cell_size; cX += cell_size){
            // Iterate through each pixel of cell (cX,cY)
            for(int y = cY; y < cY + cell_size; ++y){
                for(int x = cX; x < cX + cell_size; ++x){
                    // Lab components for each pixel
                    float l = (float)input[(z.width * y + x) * 3];
                    float a = (float)input[(z.width * y + x) * 3 + 1];
                    float b = (float)input[(z.width * y + x) * 3 + 2];

                    // Iterate trough each centroid
                    float minDist = FLT_MAX;
                    int minIdx = 0;
                    //float *inputCentroid = (float*)(_labCentroids.data);
                    for(int k = 0; k < outsize.height; ++k){
                        /*float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                + ( (a - inputCentroid[3*k + 1]) * (a - inputCentroid[3*k+1]) ) 
                                + ( (b - inputCentroid[3*k + 2]) * (b - inputCentroid[3*k+2]) );*/
                        float dist = ( (l - data1[k][0]) * (l - data1[k][0]) )
                                + ( (a - data1[k][1]) * (a - data1[k][1]) ) 
                                + ( (b - data1[k][2]) * (b - data1[k][2]) );
                        if(dist < minDist){
                            minDist = dist;
                            minIdx = k;
                        }
                    }
                    // Store result at output
                    //outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ; 
                    //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
                    out[minIdx * outsize.width + cntCell] += 1.0 / (cell_size * cell_size);
                }
            }
            cntCell++;
        }
    }
}
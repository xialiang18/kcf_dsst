#include "fhogcuda.h"
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.hpp"

extern void fhogCudaInit(int cell_size);

extern void imageGrad(unsigned char *imageData, float *dxData, float *dyData, cv::Size size);

extern void maxGrad(float *dx, float *dy, float *r, int *alfa, cv::Size size, int channels);

extern void featureMaps(float *r, int *alfa, float *map, int k, int sizeX, int sizeY, int p, cv::Size size);

extern void squareSum(float* map, float* partOfNorm, int numFeatures, int num, int size);

extern void normalization(float *map, float* partOfNorm, float * newData, cv::Size size, float alfa);

extern void PCAMaps(float *newData, float *featureData, cv::Size size, int xp, int yp);

namespace kcfcuda {

fhogFeature::fhogFeature()
{

}

fhogFeature::~fhogFeature()
{
    //cudaFree(imageData);
    cudaFree(dxData);
    cudaFree(dyData);
    cudaFree(r);
    cudaFree(alfa);
    cudaFree(d_map);
    cudaFree(d_partOfNorm);
    //cudaFree(d_newData);
    cudaFreeHost((void *)h_newData);
    cudaFree(featureData);
}

int fhogFeature::init(int imgWidth, int imgHeight, int channels, int k)
{
    cudaError_t err = cudaSuccess;
    int imageSize = imgWidth * imgHeight * channels;
    int sizeX = imgWidth / k;
    int sizeY = imgHeight / k;

    fhogCudaInit(k);

#ifdef PERFORMANCE
        Timer_Begin(cudaMalloc);
#endif

    err = cudaMallocManaged((void **)&imageData1, sizeof(char) * imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector imageData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&dxData, sizeof(float) * imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector dx (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&dyData, sizeof(float) * imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector dy (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&r, sizeof(float) * (imgWidth * imgHeight));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&alfa, sizeof(int  ) * (imgWidth * imgHeight * 2));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector alfa (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&d_map, sizeof (float) * (sizeX * sizeY  * 3 * NUM_SECTOR));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector map (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&d_partOfNorm, sizeof(float) * (sizeX * sizeY));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector partOfNorm (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //err = cudaHostAlloc((void**)&h_newData, sizeof(float) * ((sizeX - 2) * (sizeY - 2) * 12 * NUM_SECTOR), cudaHostAllocMapped);
    //err = cudaHostGetDevicePointer((void **)&d_newData, (void *)h_newData, 0);
    err = cudaMallocManaged((void**)&d_newData, sizeof(float) * ((sizeX - 2) * (sizeY - 2) * 12 * NUM_SECTOR));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector newData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void**)&featureData, sizeof(float) * ((sizeX - 2) * (sizeY - 2) * (NUM_SECTOR * 3 + 4)));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector featureData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef PERFORMANCE
        Timer_End(cudaMalloc);
#endif

    return err;
}

/*
// Getting feature map for the selected subimage
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int fhogFeature::getFeatureMaps(const IplImage *image, const int k, CvLSVMFeatureMapCaskade **map)
{
    int sizeX, sizeY;
    int p, px;
    int height, width, numChannels;

    height = image->height;
    width  = image->width ;

    sizeX = width  / k;
    sizeY = height / k;
    px    = 3 * NUM_SECTOR;
    p     = px;
    allocFeatureMapObject(map, sizeX, sizeY, p);

    numChannels = image->nChannels;
    cudaError_t err = cudaSuccess;

Timer_Begin(cudaMemcpy);
    std::cout << "size " << image->imageSize << std::endl;
    err = cudaMemcpy(imageData1, image->imageData, sizeof(char) * image->imageSize, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector getfeaturemaps1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
Timer_End(cudaMemcpy);

//#ifdef PERFORMANCE
    //Timer_Begin(imageGrad);
//#endif

    imageGrad(imageData1, dxData, dyData, cv::Size(image->widthStep, image->height));
    //cudaDeviceSynchronize();

//#ifdef PERFORMANCE
    //Timer_End(imageGrad);
//#endif

    //Timer_Begin(maxGrad);
    maxGrad(dxData, dyData, r, alfa, cv::Size(width, height), numChannels);
    //cudaDeviceSynchronize();
    //Timer_End(maxGrad);

    //Timer_Begin(cudaMemset);
    cudaMemset(d_map, 0, sizeof (float) * (sizeX * sizeY  * p));
    //cudaDeviceSynchronize();
    //Timer_End(cudaMemset);

    //Timer_Begin(featureMaps);
    featureMaps(r, alfa, d_map, k, sizeX, sizeY, p, cv::Size(width, height));
    //cudaDeviceSynchronize();
    //Timer_End(featureMaps);

    return LATENT_SVM_OK;
}

/*
// Feature map Normalization and Truncation
//
// API
// int normalizeAndTruncate(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
int fhogFeature::normalizeAndTruncate(CvLSVMFeatureMapCaskade *map, const float alfa)
{
    int sizeX, sizeY, p, pp, xp;
    //float * newData;

    sizeX     = map->sizeX;
    sizeY     = map->sizeY;

    p  = NUM_SECTOR;
    xp = NUM_SECTOR * 3;
    pp = NUM_SECTOR * 12;

//Timer_Begin(squareSum);
    squareSum(d_map, d_partOfNorm, xp, p, sizeX * sizeY);
    //cudaDeviceSynchronize();
//Timer_End(squareSum);

    sizeX -= 2;
    sizeY -= 2;

    //newData = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));

//Timer_Begin(normalization);
    normalization(d_map, d_partOfNorm, d_newData, cv::Size(sizeX, sizeY), alfa);
    //cudaDeviceSynchronize();
    
//Timer_End(normalization);

    //cudaMemcpy(newData, d_newData, sizeof(float) * (sizeX * sizeY * pp), cudaMemcpyDeviceToHost);

    map->numFeatures  = pp;
    map->sizeX = sizeX;
    map->sizeY = sizeY;

    //free (map->map);

    map->map = d_newData;

    return LATENT_SVM_OK;
}
/*
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int fhogFeature::PCAFeatureMaps(CvLSVMFeatureMapCaskade *map)
{
    int i,j, ii, jj, k;
    int sizeX, sizeY, p,  pp, xp, yp, pos1, pos2;
    float * newData;
    float val;
    float nx, ny;

    sizeX = map->sizeX;
    sizeY = map->sizeY;
    p     = map->numFeatures;
    pp    = NUM_SECTOR * 3 + 4;
    yp    = 4;
    xp    = NUM_SECTOR;

    nx    = 1.0f / sqrtf((float)(xp * 2));
    ny    = 1.0f / sqrtf((float)(yp    ));

    //newData = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));
    //Timer_Begin(PCAFeatureMaps);

    PCAMaps(d_newData, featureData, cv::Size(sizeX, sizeY), xp, yp);

    cudaDeviceSynchronize();

    //cudaMemcpy(newData, featureData, sizeof(float) * (sizeX * sizeY * pp), cudaMemcpyDeviceToHost);

    //Timer_End(PCAFeatureMaps);
//swop data
    map->numFeatures = pp;

    //free (map->map);

    map->map = featureData;

    return LATENT_SVM_OK;
}


//modified from "lsvmc_routine.cpp"

int fhogFeature::allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX,
                          const int sizeY, const int numFeatures)
{
    int i;
    (*obj) = (CvLSVMFeatureMapCaskade *)malloc(sizeof(CvLSVMFeatureMapCaskade));
    (*obj)->sizeX       = sizeX;
    (*obj)->sizeY       = sizeY;
    (*obj)->numFeatures = numFeatures;
    // (*obj)->map = (float *) malloc(sizeof (float) *
    //                               (sizeX * sizeY  * numFeatures));
    // for(i = 0; i < sizeX * sizeY * numFeatures; i++)
    // {
    //     (*obj)->map[i] = 0.0f;
    // }
    return LATENT_SVM_OK;
}

int fhogFeature::freeFeatureMapObject (CvLSVMFeatureMapCaskade **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    //free((*obj)->map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

}

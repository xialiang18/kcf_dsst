/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
//#include "fhog.hpp"
#include "labdata.hpp"
#include "timer.hpp"
#endif

#include <cuda_runtime.h>

extern void getLabFeatures(cv::Size z, int cell_size, unsigned char *input, float *out, cv::Size outsize);

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5;
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;

    is_trusted = true;
    pv_threshold = 0.6;
    apce_threshold = 0.45;
    pv_max = 0;
    apce_max = 0;

    if (hog) {    // HOG
        // VOT
        interp_factor = 0.012;
        sigma = 0.6;
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5;
        cell_size = 4;
        _hogfeatures = true;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4;
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }


    if (multiscale) { // multiscale
        template_size = 96;
        //scale parameters initial
        scale_padding = 1.0;
        scale_step = 1.05;
        scale_sigma_factor = 0.25;
        n_scales = 33;
        scale_lr = 0.025;
        scale_max_area = 512;
        currentScaleFactor = 1;
        scale_lambda = 0.01;

        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }

}

// Initialize tracker
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(size_patch[1][0], size_patch[1][1]);
    _alphaf = cv::Mat(size_patch[1][0], size_patch[1][1], CV_32FC2, float(0));

    dsstInit(roi, image);
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame
 }

// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
#ifdef PERFORMANCE
        Timer_Begin(update);
#endif

    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

#ifdef PERFORMANCE
        Timer_Begin(detect);
#endif

    
    //KCFTracker::SEARCH_METHOD method = is_trusted ? KCFTracker::PART_SEARCH : KCFTracker::ALL_SEARCH;
    KCFTracker::SEARCH_METHOD method = KCFTracker::PART_SEARCH;
//    Timer_Begin(feature3);
    cv::Mat feature = getFeatures(image, 0, method);
//    Timer_End(feature3);
    cv::Point2f res = detect(_tmpl, feature, method);

#ifdef PERFORMANCE
        Timer_End(detect);
#endif
    
    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale * currentScaleFactor);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale * currentScaleFactor);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;


/*    // Update scale
#ifdef PERFORMANCE
        Timer_Begin(detect_scale);
#endif

    cv::Point2i scale_pi = detect_scale(image);

#ifdef PERFORMANCE
        Timer_End(detect_scale);
#endif
    currentScaleFactor = currentScaleFactor * scaleFactors[scale_pi.x];*/
    /*std::ofstream fout("scale_result.txt", std::ios::app);
    fout << currentScaleFactor << std::endl;*/
/*    if(currentScaleFactor < min_scale_factor)
      currentScaleFactor = min_scale_factor;
    std::cout << currentScaleFactor << std::endl;
    // else if(currentScaleFactor > max_scale_factor)
    //   currentScaleFactor = max_scale_factor;

#ifdef PERFORMANCE
        Timer_Begin(train_scale);
#endif

    train_scale(image);

#ifdef PERFORMANCE
        Timer_End(train_scale);
#endif

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
*/
#ifdef PERFORMANCE
        Timer_Begin(getFeatures);
#endif

    assert(_roi.width >= 0 && _roi.height >= 0);

    if(is_trusted){
        //Timer_Begin(feature);
        cv::Mat x = getFeatures(image, 0);
        //Timer_End(feature);

    #ifdef PERFORMANCE
            Timer_End(getFeatures);
    #endif

    #ifdef PERFORMANCE
            Timer_Begin(train);
    #endif

        train(x, interp_factor);

    #ifdef PERFORMANCE
            Timer_End(train);
    #endif
    }
    #ifdef PERFORMANCE
            Timer_End(update);
    #endif
    return _roi;
}

// Detect the new scaling rate
cv::Point2i KCFTracker::detect_scale(cv::Mat image)
{
  cv::Mat xsf = KCFTracker::get_scale_sample(image);

  // Compute AZ in the paper
  cv::Mat add_temp;
  cv::reduce(FFTTools::complexMultiplication(sf_num, xsf), add_temp, 0, CV_REDUCE_SUM);

  // compute the final y
  cv::Mat scale_response;
  cv::idft(FFTTools::complexDivisionReal(add_temp, (sf_den + scale_lambda)), scale_response, cv::DFT_REAL_OUTPUT);

  // Get the max point as the final scaling rate
  cv::Point2i pi;
  double pv;
  double pmin;
  cv::minMaxLoc(scale_response, &pmin, &pv, NULL, &pi);
  
    //std::cout << pv << std::endl;
  return pi;
}


// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, KCFTracker::SEARCH_METHOD method)
{
    using namespace FFTTools;

    //static int first_time = 1;
    static int cnt = 0;
    cv::Mat res;
    float peak_value, min_value;
    cv::Point2f p = gaussianCorrelation(x, z, method, res, peak_value, min_value);

    /*static int num = 0;
    num++;
    char res_graph[50];
    sprintf(res_graph, "corr/corr_%d.txt", num);
    std::ofstream corr_fout(res_graph);
    for(int i = 0; i < res.rows; i++)
        for(int j = 0; j < res.cols; j++){
            corr_fout << i << " " << j << " " << res.at<float>(i, j) << std::endl;
        }

    corr_fout.close();*/

    res = res - min_value;
    res = res.mul(res);
    cv::Scalar tempval = cv::mean(res);
    float mean = tempval.val[0];
    float apce = std::pow(peak_value - min_value, 2.0) / mean;
    /*std::ofstream fout("c_result.txt", std::ios::app);
    fout << num << " " << peak_value << " " << apce << std::endl;
    fout.close();*/

    if((peak_value >= pv_max * pv_threshold) && (apce >= apce_max * apce_threshold)){
        is_trusted = true;
        pv_max = (pv_max * cnt + peak_value) / (float)(cnt + 1);
        apce_max = (apce_max * cnt + apce) / (float)(cnt + 1);
        cnt++;
    }else{
        is_trusted = false;
    }

    return p;
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianAutocorrelation(x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;


    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Point2f KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2, KCFTracker::SEARCH_METHOD method, cv::Mat &res, float &peak_value, float &min_value)
{
    using namespace FFTTools;

    int width = size_patch[1][1];
    int height = size_patch[1][0];
    int total_width = size_patch[method][1];
    int total_height = size_patch[method][0];
    
    peak_value = 0;
    int search_step_width = 2;
    int search_step_height = 2;
    cv::Point2f res_point;
    cv::Point2i pi;
    for(int m = 0; m <= total_height - height; m = m + search_step_height){
        for(int n = 0; n <= total_width - width; n = n + search_step_width){
            cv::Mat c = cv::Mat( cv::Size(size_patch[1][1], size_patch[1][0]), CV_32F, cv::Scalar(0) );
            // HOG features
            cv::Mat caux;
            cv::Mat x1aux;
            cv::Mat x2aux;
            //int width_begin = n * width;
            //int height_begin = m * height;
            for (int i = 0; i < size_patch[1][2]; i++) {
                x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
                x1aux = x1aux.reshape(1, size_patch[method][0]);
                x1aux = x1aux.rowRange(m, m + height).colRange(n, n + width);
                x2aux = x2.row(i).reshape(1, size_patch[1][0]);
                //std::cout << "rows " << x1aux.rows << " " << x2aux.rows << " cols " << x1aux.cols << " " << x2aux.cols << std::endl;
                cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true); 
                caux = fftd(caux, true);
                rearrange(caux);
                caux.convertTo(caux, CV_32F);
                c = c + real(caux);
            }

            cv::Mat d; 
            cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[1][0] * size_patch[1][1] * size_patch[1][2]) , 0, d);

            cv::Mat k;
            cv::exp((-d / (sigma * sigma)), k);

            cv::Mat res_temp = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
            double pv;
            double mv;
            cv::minMaxLoc(res_temp, &mv, &pv, NULL, &pi);

            if(pv > peak_value){
                peak_value = pv;
                min_value = mv;
                res_point.x = n + pi.x;
                res_point.y = m + pi.y;
                res = res_temp;
            }
        }
    }

    if (res_point.x > 0 && res_point.x < res.cols-1) {
        res_point.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
    }

    if (res_point.y > 0 && res_point.y < res.rows-1) {
        res_point.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
    }

    res_point.x -= (total_width) / 2;
    res_point.y -= (total_height) / 2;

    return res_point;
}

cv::Mat KCFTracker::gaussianAutocorrelation(const cv::Mat x)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat( cv::Size(size_patch[1][1], size_patch[1][0]), CV_32F, cv::Scalar(0) );
    cv::Mat caux;
    cv::Mat x1aux;
    cv::Mat x2aux;
    for (int i = 0; i < size_patch[1][2]; i++) {
        x1aux = x.row(i);   // Procedure do deal with cv::Mat multichannel bug
        x1aux = x1aux.reshape(1, size_patch[1][0]);
        x2aux = x.row(i).reshape(1, size_patch[1][0]);
        cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true); 
        caux = fftd(caux, true);
        rearrange(caux);
        caux.convertTo(caux,CV_32F);
        c = c + real(caux);
    }

    cv::Mat d; 
    cv::max(( (cv::sum(x.mul(x))[0] + cv::sum(x.mul(x))[0])- 2. * c) / (size_patch[1][0] * size_patch[1][1] * size_patch[1][2]) , 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, KCFTracker::SEARCH_METHOD method)
{
    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;


    if (inithann) {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;
        
        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz[1].width = padded_w / _scale;
            _tmpl_sz[1].height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size
            _tmpl_sz[1].width = padded_w;
            _tmpl_sz[1].height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }

        if (_hogfeatures) {
            // Round to cell size and also make it even
            _tmpl_sz[1].width = ( ( (int)(_tmpl_sz[1].width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
            _tmpl_sz[1].height = ( ( (int)(_tmpl_sz[1].height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
        }
        else {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz[1].width = (_tmpl_sz[1].width / 2) * 2;
            _tmpl_sz[1].height = (_tmpl_sz[1].height / 2) * 2;
        }
    }
    /*static int size_init = 1;
    static int update_flag = 1;
    if(inithann == 0 && size_init && update_flag){
        
        size_init = 0;
    }*/

    if(_tmpl_sz.find(method) == _tmpl_sz.end()){
        if(method == KCFTracker::PART_SEARCH){
            _tmpl_sz[method].width = _tmpl_sz[1].width * method;
            _tmpl_sz[method].height = _tmpl_sz[1].height * method;
        }else if(method == KCFTracker::ALL_SEARCH){
            int image_width = image.cols;
            int image_height = image.rows;
            /*int unit_width = _scale * _tmpl_sz[1].width;
            int unit_height = _scale * _tmpl_sz[1].height;
            int width_scale = image_width / unit_width;
            int height_scale = image_height / unit_height;
            _tmpl_sz[method].width = _tmpl_sz[1].width * width_scale;
            _tmpl_sz[method].height = _tmpl_sz[1].height * height_scale;*/
            int temp_width = image_width / _scale;
            int temp_height = image_height / _scale;
            _tmpl_sz[method].width = temp_width / (2 * cell_size) * (2 * cell_size);
            _tmpl_sz[method].height = temp_height / (2 * cell_size) * (2 * cell_size);
        }
    }

    extracted_roi.width = _scale * _tmpl_sz[method].width;
    extracted_roi.height = _scale * _tmpl_sz[method].height;
    //std::cout << extracted_roi.width << " " << extracted_roi.height << std::endl;

    if(method == KCFTracker::ALL_SEARCH){
        cx = image.cols / 2;
        cy = image.rows / 2;
    }

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

//#ifdef PERFORMANCE
        //Timer_Begin(subwindow);
//#endif
    cv::Mat FeaturesMap;  
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    
    //Timer_End(subwindow);

    //Timer_Begin(resize);
    if (z.cols != _tmpl_sz[method].width || z.rows != _tmpl_sz[method].height) {
        cv::resize(z, z, _tmpl_sz[method]);
    }   
//#ifdef PERFORMANCE
     //   Timer_End(resize);
//#endif

    // HOG features
    if (_hogfeatures) {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
        if(fhog.find(method) == fhog.end()){
            fhog[method] = new kcfcuda::fhogFeature();
            fhog[method]->init(z_ipl.width, z_ipl.height, z_ipl.nChannels, cell_size);
        }
//#ifdef PERFORMANCE
        Timer_Begin(fhog);
//#endif
Timer_Begin(getFeatureMaps);
        fhog[method]->getFeatureMaps(&z_ipl, cell_size, &map);
Timer_End(getFeatureMaps);
Timer_Begin(normalizeAndTruncate);
        fhog[method]->normalizeAndTruncate(map,0.2f);
Timer_End(normalizeAndTruncate);
Timer_Begin(PCAFeatureMaps);
        fhog[method]->PCAFeatureMaps(map);
Timer_End(PCAFeatureMaps);
//#ifdef PERFORMANCE
        Timer_End(fhog);
//#endif
        if(size_patch.find(method) == size_patch.end()){
            size_patch[method].emplace_back(map->sizeY);
            size_patch[method].emplace_back(map->sizeX);
            size_patch[method].emplace_back(map->numFeatures + _labCentroids.rows);
        }
        
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map);
        FeaturesMap = FeaturesMap.t();
        fhog[method]->freeFeatureMapObject(&map);

//#ifdef PERFORMANCE
    //    Timer_Begin(lab);
//#endif
        // Lab features
        if (_labfeatures) {
            cv::Mat imgLab;
            //Timer_Begin(cvtColor);
            cvtColor(z, imgLab, CV_BGR2Lab);
            //Timer_End(cvtColor);
            unsigned char *input = (unsigned char*)(imgLab.data);

            //Timer_Begin(change);
            // Iterate through each cell
            if(lab_input_data.find(method) == lab_input_data.end())
                cudaMallocManaged((void **)&(lab_input_data[method]), sizeof(unsigned char) * imgLab.rows * imgLab.cols * imgLab.channels());
            cudaError_t err = cudaMemcpyAsync(lab_input_data[method], input, sizeof(unsigned char) * imgLab.rows * imgLab.cols * imgLab.channels(), cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                fprintf(stderr, "Failed to allocate device vector getfeaturemaps1 (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //float *lab_data;
            if(lab_feature_data.find(method) == lab_feature_data.end())
                cudaMallocManaged((void **)&(lab_feature_data[method]), sizeof(float) * _labCentroids.rows * size_patch[method][0] * size_patch[method][1]);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector partOfNorm (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            getLabFeatures(cv::Size(z.cols, z.rows), cell_size, lab_input_data[method], lab_feature_data[method], cv::Size(size_patch[method][0] * size_patch[method][1], _labCentroids.rows));

            cv::Mat outputLab = cv::Mat(_labCentroids.rows, 
                                        size_patch[method][0] * size_patch[method][1], 
                                        CV_32F, 
                                        lab_feature_data[method]);

            //Timer_End(change);
            FeaturesMap.push_back(outputLab);
        }

//#ifdef PERFORMANCE
        //Timer_End(lab);
//#endif
    }
    else {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;
        /*size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;*/
        size_patch[method].emplace_back(z.rows);
        size_patch[method].emplace_back(z.cols);
        size_patch[method].emplace_back(1);
    }
    
    if (hann.find(method) == hann.end()) {
        createHanningMats(method);
    }

    if(method != KCFTracker::ALL_SEARCH)
        FeaturesMap = hann[method].mul(FeaturesMap);

    return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats(KCFTracker::SEARCH_METHOD method)
{   
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[method][1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[method][0]), CV_32F, cv::Scalar(0)); 

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
        hann[method] = cv::Mat( cv::Size(size_patch[method][0] * size_patch[method][1], 
                        size_patch[method][2]), 
                        CV_32F, 
                        cv::Scalar(0) );

        for (int i = 0; i < size_patch[method][2]; i++) {
            for (int j = 0; j < size_patch[method][0] * size_patch[method][1]; j++) {
                hann[method].at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
    // Gray features
    else {
        hann[method] = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}

// Initialization for scales
void KCFTracker::dsstInit(const cv::Rect &roi, cv::Mat image)
{
  // The initial size for adjusting
  base_width = roi.width;
  base_height = roi.height;

  // Guassian peak for scales (after fft)
  ysf = computeYsf();
  s_hann = createHanningMatsForScale();

  // Get all scale changing rate
  scaleFactors = new float[n_scales];
  float ceilS = std::ceil(n_scales / 2.0f);
  for(int i = 0 ; i < n_scales; i++)
  {
    scaleFactors[i] = std::pow(scale_step, ceilS - i - 1);
  }

  // Get the scaling rate for compressing to the model size
  float scale_model_factor = 1;
  if(base_width * base_height > scale_max_area)
  {
    scale_model_factor = std::sqrt(scale_max_area / (float)(base_width * base_height));
  }
  scale_model_width = (int)(base_width * scale_model_factor);
  scale_model_height = (int)(base_height * scale_model_factor);

  // Compute min and max scaling rate
  min_scale_factor = std::pow(scale_step,
    std::ceil(std::log((std::fmax(5 / (float) base_width, 5 / (float) base_height) * (1 + scale_padding))) / 0.0086));
  max_scale_factor = std::pow(scale_step,
    std::floor(std::log(std::fmin(image.rows / (float) base_height, image.cols / (float) base_width)) / 0.0086));

  train_scale(image, true);

}

// Train method for scaling
void KCFTracker::train_scale(cv::Mat image, bool ini)
{
  cv::Mat xsf = get_scale_sample(image);

  // Adjust ysf to the same size as xsf in the first time
  if(ini)
  {
    int totalSize = xsf.rows;
    ysf = cv::repeat(ysf, totalSize, 1);
  }

  // Get new GF in the paper (delta A)
  cv::Mat new_sf_num;
  cv::mulSpectrums(ysf, xsf, new_sf_num, 0, true);

  // Get Sigma{FF} in the paper (delta B)
  cv::Mat new_sf_den;
  cv::mulSpectrums(xsf, xsf, new_sf_den, 0, true);
  cv::reduce(FFTTools::real(new_sf_den), new_sf_den, 0, CV_REDUCE_SUM);

  if(ini)
  {
    sf_den = new_sf_den;
    sf_num = new_sf_num;
  }else
  {
    // Get new A and new B
    cv::addWeighted(sf_den, (1 - scale_lr), new_sf_den, scale_lr, 0, sf_den);
    cv::addWeighted(sf_num, (1 - scale_lr), new_sf_num, scale_lr, 0, sf_num);
  }

  update_roi();

}

// Update the ROI size after training
void KCFTracker::update_roi()
{
  // Compute new center
  float cx = _roi.x + _roi.width / 2.0f;
  float cy = _roi.y + _roi.height / 2.0f;

  // printf("%f\n", currentScaleFactor);

  // Recompute the ROI left-upper point and size
  _roi.width = base_width * currentScaleFactor;
  _roi.height = base_height * currentScaleFactor;

  _roi.x = cx - _roi.width / 2.0f;
  _roi.y = cy - _roi.height / 2.0f;

}

// Compute the F^l in the paper
cv::Mat KCFTracker::get_scale_sample(const cv::Mat & image)
{
  CvLSVMFeatureMapCaskade *map[n_scales]; // temporarily store FHOG result
  cv::Mat xsf; // output
  int totalSize; // # of features

  for(int i = 0; i < n_scales; i++)
  {
    // Size of subwindow waiting to be detect
    float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
    float patch_height = base_height * scaleFactors[i] * currentScaleFactor;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    // Get the subwindow
    cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
    cv::Mat im_patch_resized;

    // Scaling the subwindow
    if(scale_model_width > im_patch.cols)
      resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 1);
    else
      resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 3);

    // Compute the FHOG features for the subwindow
    IplImage im_ipl = im_patch_resized;
    fhog[1]->getFeatureMaps(&im_ipl, cell_size, &map[i]);
    fhog[1]->normalizeAndTruncate(map[i], 0.2f);
    fhog[1]->PCAFeatureMaps(map[i]);

    if(i == 0)
    {
      totalSize = map[i]->numFeatures*map[i]->sizeX*map[i]->sizeY;
      xsf = cv::Mat(cv::Size(n_scales,totalSize), CV_32F, float(0));
    }

    // Multiply the FHOG results by hanning window and copy to the output
    cv::Mat FeaturesMap = cv::Mat(cv::Size(1, totalSize), CV_32F, map[i]->map);
    float mul = s_hann.at<float > (0, i);
    FeaturesMap = mul * FeaturesMap;
    FeaturesMap.copyTo(xsf.col(i));

  }

  // Free the temp variables
  for(int i = 0; i < n_scales; i++)
      fhog[1]->freeFeatureMapObject(&map[i]);

  // Do fft to the FHOG features row by row
  xsf = FFTTools::fftd(xsf, 0, 1);

  return xsf;
}

// Compute the FFT Guassian Peak for scaling
cv::Mat KCFTracker::computeYsf()
{
    float scale_sigma2 = n_scales / std::sqrt(n_scales) * scale_sigma_factor;
    scale_sigma2 = scale_sigma2 * scale_sigma2;
    cv::Mat res(cv::Size(n_scales, 1), CV_32F, float(0));
    float ceilS = std::ceil(n_scales / 2.0f);

    for(int i = 0; i < n_scales; i++)
    {
      res.at<float>(0,i) = std::exp(- 0.5 * std::pow(i + 1- ceilS, 2) / scale_sigma2);
    }

    return FFTTools::fftd(res);

}

// Compute the hanning window for scaling
cv::Mat KCFTracker::createHanningMatsForScale()
{
  cv::Mat hann_s = cv::Mat(cv::Size(n_scales, 1), CV_32F, cv::Scalar(0));
  for (int i = 0; i < hann_s.cols; i++)
      hann_s.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann_s.cols - 1)));

  return hann_s;
}

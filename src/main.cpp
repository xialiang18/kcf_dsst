#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <kcftracker.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <timer.hpp>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

 /* std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve time cost " << time_used.count() << " seconds." << std::endl;
*/

//#ifdef PERFORMANCE
    std::map<std::string, std::vector<std::chrono::microseconds>> TimeLogger::durations_;
//#endif

#define max(a, b) (a) > (b) ? (a) : (b)
#define min(a, b) (a) < (b) ? (a) : (b)

float calc_iou(cv::Rect res_roi, cv::Rect gt_roi)
{
    float iou;
    float resarea, gtarea, overlap;
    resarea = res_roi.width * res_roi.height;
    gtarea = gt_roi.width * gt_roi.height;
    float lx, ly, rx, ry;
    lx = max(res_roi.x, gt_roi.x);
    ly = max(res_roi.y, gt_roi.y);
    rx = min(res_roi.x + res_roi.width, gt_roi.x + gt_roi.width);
    ry = min(res_roi.y + res_roi.height, gt_roi.y + gt_roi.height);
    overlap = (rx - lx) * (ry - ly);
    iou = overlap / (resarea + gtarea - overlap + 1e-6);
    return iou;
}

void verify_result(std::string result, std::string gt, float threshold)
{
    std::fstream resfile(result), gtfile(gt);

    std::string resline, gtline;
    cv::Rect_<float> res_roi, gt_roi;
    int cnt = 0;
    while(getline(resfile, resline)){
        getline(gtfile, gtline);
        std::istringstream resstream(resline), gtstream(gtline);
        float resdata[4], gtdata[4];
        char s;
        for(int i = 0; i < 4; i++){
            resstream >> resdata[i];
            resstream >> s;
            gtstream >> gtdata[i];
            gtstream >> s;
        }

        res_roi.x = resdata[0];
        res_roi.y = resdata[1];
        res_roi.width = resdata[2];
        res_roi.height = resdata[3];

        gt_roi.x = gtdata[0];
        gt_roi.y = gtdata[1];
        gt_roi.width = gtdata[2];
        gt_roi.height = gtdata[3];

        float iou;
        float resarea, gtarea, overlap;
        resarea = res_roi.width * res_roi.height;
        gtarea = gt_roi.width * gt_roi.height;
        float lx, ly, rx, ry;
        lx = max(res_roi.x, gt_roi.x);
        ly = max(res_roi.y, gt_roi.y);
        rx = min(res_roi.x + res_roi.width, gt_roi.x + gt_roi.width);
        ry = min(res_roi.y + res_roi.height, gt_roi.y + gt_roi.height);
        overlap = (rx - lx) * (ry - ly);
        iou = overlap / (resarea + gtarea - overlap + 1e-6);
        cnt++;
        if(iou < threshold){
            std::cout << "track failed!" << std::endl;
        }else{
            std::cout << "Test " << cnt << " track success!" << std::endl;
        }
    }

    resfile.close();
    gtfile.close();
}

extern int test(void);

cv::Mat move_image(cv::Mat image, int x, int y){
    int width = image.cols;
    int height = image.rows;
    cv::Mat a = image.colRange(0, (width - x) % width);
    cv::Mat b = image.colRange((width - x) % width, width);
    cv::Mat temp_x = b.t();
    a = a.t();
    temp_x.push_back(a);
    temp_x = temp_x.t();
    a = temp_x.rowRange(0, (height - y) % height);
    b = temp_x.rowRange((height - y) % height, height);
    b.push_back(a);
    return b;
}

void data_generate()
{
    srand((unsigned int)time(0));

    for(int length = 20; length < 420; length += 20){
        int x = 496, y = 419, width = 40, height = 42;
        int cnt = 0;
        cv::Mat frame = cv::imread("data/ball1/00000001.jpg");
        int frame_width = frame.cols;
        int frame_height = frame.rows;
        char path[50];
        sprintf(path, "data/dist/dist_%d", length);
        mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        char gt[50];
        sprintf(gt, "data/dist/dist_%d/gt.txt", length);
        std::ofstream fout(gt, std::ios::app);
        while(cnt < 1000){
            float theta = (double)rand() / RAND_MAX * 2 * 3.141592654;
            int dx = (int)(length * cos(theta));
            int dy = (int)(length * sin(theta));
            if(x + dx >= 0 && x + dx + width < frame_width && y + dy >= 0 && y + dy + height < frame_height){
                cnt++;
                frame = move_image(frame, dx, dy);
                x = x + dx;
                y = y + dy;
                //std::string data = "data/dist/dist_";
                //data += length + "/" + cnt + ".jpg";
                char data[100];
                sprintf(data, "data/dist/dist_%d/%d.jpg", length, cnt);
                //sprintf(data, "")
                cv::imwrite(data, frame);
                fout << x << "," << y << "," << width << "," << height << std::endl;
            }
        }
    }
}

void test_search()
{

    Tracker *tracker = new KCFTracker();
    
    for(int length = 20; length < 420; length += 20){
        std::cout << "running " << length << std::endl;
        char path[50], gt[50];
        
        sprintf(gt, "data/dist/dist_%d/gt.txt", length);
        std::ifstream fin(gt);
        std::string ini;
        std::getline(fin, ini);
        fin.close();
        std::istringstream temp(ini);
        char s;
        cv::Rect roi;
        temp >> roi.x >> s >> roi.y >> s >> roi.width >> s >> roi.height;
        cv::Mat frame;
        char outfile[50];
        sprintf(outfile, "data/dist/dist_%d/result.txt", length);
        std::ofstream fout(outfile);
        int initFlag = 1;
        for(int i = 1; i <= 1000; i++){
            sprintf(path, "data/dist/dist_%d/%d.jpg", length, i);
            frame = cv::imread(path);
            if(initFlag)
                tracker->init( roi, frame );
            else{
                roi = tracker->update(frame);
            }
            fout << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
            initFlag = 0;
        }
        fout.close();


        
        char result[50];
        sprintf(result, "test_res/result_%d.txt", length);
        std::ofstream ff(result);
        std::vector<float> res_iou;
        std::ifstream gt_i(gt), res_i(outfile);
        std::string res_line, gt_line;
        cv::Rect gt_p, res_p;
        float dist = 0;
        while(getline(res_i, res_line) && getline(gt_i, gt_line)){
            std::istringstream res_temp(res_line);
            std::istringstream gt_temp(gt_line);
            char s;
            res_temp >> res_p.x >> s >> res_p.y >> s >> res_p.width >> s >> res_p.height;
            gt_temp >> gt_p.x >> s >> gt_p.y >> s >> gt_p.width >> s >> gt_p.height;
            //dist += std::sqrt(std::pow(res_p.x - gt_p.x, 2.0) + std::pow(res_p.y - gt_p.y, 2.0)) / 1000.0;
            float iou = calc_iou(res_p, gt_p);
            res_iou.push_back(iou);
        }
        gt_i.close();
        res_i.close();

        for(float base = 0.1; base <= 1.001; base += 0.1){
            dist = 0;
            for(auto v : res_iou){
                if(v >= base){
                    dist++;
                }
            }
            dist /= 1000.0;
            ff << base << " " << dist << std::endl;
        }
        ff.close();
    }
}

void test_scale()
{
    for(int i = 0; i < 2; i++){
        char result[50], gt[50], outfile[50];
        if(i == 0){
            sprintf(result, "res/result_scale.txt");
            sprintf(outfile, "res/result_scale_1.txt");
        }else{
            sprintf(result, "res/result.txt");
            sprintf(outfile, "res/result_1.txt");
        }
        sprintf(gt, "res/groundtruth_rect.txt");
        std::ofstream ff(outfile);
        std::vector<float> res_iou;
        std::ifstream gt_i(gt), res_i(result);
        std::string res_line, gt_line;
        cv::Rect gt_p, res_p;
        float dist = 0;
        int cnt = 0;
        while(getline(res_i, res_line) && getline(gt_i, gt_line)){
            cnt++;
            std::istringstream res_temp(res_line);
            std::istringstream gt_temp(gt_line);
            char s;
            res_temp >> res_p.x >> s >> res_p.y >> s >> res_p.width >> s >> res_p.height;
            gt_temp >> gt_p.x >> s >> gt_p.y >> s >> gt_p.width >> s >> gt_p.height;
            //dist += std::sqrt(std::pow(res_p.x - gt_p.x, 2.0) + std::pow(res_p.y - gt_p.y, 2.0)) / 1000.0;
            float iou = calc_iou(res_p, gt_p);
            res_iou.push_back(iou);
        }
        gt_i.close();
        res_i.close();

        for(float base = 0.1; base <= 1.001; base += 0.05){
            dist = 0;
            for(auto v : res_iou){
                if(v >= base){
                    dist++;
                }
            }
            dist /= (float)cnt;
            ff << base << " " << dist << std::endl;
        }
        ff.close();
    }
}

int main()
{
    std::stringstream photoName;
    //std::string picture_path = "data/ball1/";
    //std::string picture_path = "data/cat/";
    //std::string picture_path = "data/CarScale/img/";
    //std::string picture_path = "data/dog/";
    std::string picture_path = "data/FaceOcc1/img/";
    std::string result_file = "res/result.txt";
    int num = 100;
    int initFlag = 1;
    //cv::Rect roi(556, 203, 222, 209);
    //cv::Rect roi(496, 419, 40, 42);
    //cv::Rect roi(6, 166, 42, 26);
    //cv::Rect roi(205, 218, 69, 40);
    cv::Rect roi(118, 69, 114, 162);

    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool LAB = true;
    Tracker *tracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    std::ofstream result(result_file);

    cv::Mat frame;

    for(int i = 1; i <= num; i++){
        photoName.clear();
        photoName << picture_path << std::setfill('0') << std::setw(4) << i << ".jpg";
        //photoName << picture_path << i << ".jpg";
        frame = cv::imread(photoName.str().c_str());
        if(frame.empty()){
            std::cout << "frame read error!" << std::endl;
            break;
        }

#ifdef PERFORMANCE
        Timer_Begin(main);
#endif
        if(initFlag)
            tracker->init( roi, frame );
        else{
            roi = tracker->update(frame);
        }
#ifdef PERFORMANCE
        Timer_End(main);
#endif

        cv::rectangle( frame, cv::Point( roi.x, roi.y ), cv::Point( roi.x + roi.width, roi.y + roi.height), cv::Scalar( 0, 255, 255 ), 1, 8 );

#ifdef RESULT_SAVE_PHOTO
        char result_photo[50];
        sprintf(result_photo, "res/photo/%d.jpg", i);
        cv::imwrite(result_photo, frame);
#endif

        result << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
        initFlag = 0;
        photoName.str("");
    }
    result.close();

#ifdef PRECISION
    std::string gt_file = "res/gt.txt";
    verify_result(result_file, gt_file, 0.7);
#endif

    //test();
    
    return 0;
}
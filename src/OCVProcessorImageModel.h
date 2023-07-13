//
//  OCVProcessorImageModel.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef IOS_SIMULATOR
#ifndef OCVProcessorImageModel_h
#define OCVProcessorImageModel_h

#import "OCVContour.h"
#import "OCVProcessorSettings.h"
#import <opencv2/opencv2.h>

class OCVProcessorImageModel {
public:
    std::string key;
    std::vector<cv::Mat> imageMats; // TODO: can we use single image to detect?
    cv::Mat originalImageMat;
    std::vector<OCVContour> ocvContours;
    
    OCVProcessorImageModel(std::string key, cv::Mat originalImageMat, OCVProcessorSettings settings) {
        this->key = key;
        this->originalImageMat = originalImageMat;
        
        for (int i = 0; i < settings.scales.size(); i++) {
            float scale = settings.scales[i];
            GLfloat k = (settings.minVideoSize * scale) / originalImageMat.cols;
            
            cv::Mat resizedMat;
            resize(originalImageMat, resizedMat, cv::Size(originalImageMat.cols * k, originalImageMat.rows * k), INTER_LINEAR);
            cv::cvtColor(resizedMat, resizedMat, COLOR_BGR2GRAY);
            imageMats.insert(imageMats.end(), resizedMat);
        }
        
        cv::cvtColor(originalImageMat, originalImageMat, COLOR_BGR2GRAY);
        cv::threshold(originalImageMat, originalImageMat, 100, 255, THRESH_BINARY_INV);
        //        cv::Canny(originalImageMat, originalImageMat, 100,200);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(originalImageMat, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        ocvContours = convertContours(contours);
    }
    
};

#endif // OCVProcessorImageModel_h
#endif // IOS_SIMULATOR

//
//  OCVProcessorImageModel.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef OCVProcessorImageModel_h
#define OCVProcessorImageModel_h

#import "OCVContour.h"
#import "OCVProcessorSettings.h"

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
            float k = (settings.minVideoSize * scale) / originalImageMat.cols;
            
            cv::Mat resizedMat;
            resize(originalImageMat, resizedMat, cv::Size(originalImageMat.cols * k, originalImageMat.rows * k), cv::INTER_LINEAR);
            cv::cvtColor(resizedMat, resizedMat, cv::COLOR_BGR2GRAY);
            imageMats.insert(imageMats.end(), resizedMat);
        }
        
        cv::cvtColor(originalImageMat, originalImageMat, cv::COLOR_BGR2GRAY);
        cv::threshold(originalImageMat, originalImageMat, 100, 255, cv::THRESH_BINARY_INV);
        //        cv::Canny(originalImageMat, originalImageMat, 100,200);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(originalImageMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        ocvContours = convertContours(contours);
    }
    
};

#endif // OCVProcessorImageModel_h

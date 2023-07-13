//
//  OCVProcessor.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef IOS_SIMULATOR
#ifndef OCVProcessor_h
#define OCVProcessor_h

#ifdef OCV_PRINTLOGS
#define OCVLog(FORMAT, ...) NSLog(@FORMAT, ##__VA_ARGS__)
#else
#define OCVLog(FORMAT, ...) { }
#endif

#import "OCVProcessorImageModel.h"
#import "OCVProcessorSettings.h"
#import "OCVProcessorResult.h"
#import <opencv2/opencv2.h>
class OCVItemPrivateResult;
class OCVPrivateResult;

class OCVProcessor {
    std::vector<OCVProcessorImageModel> models;
    OCVProcessorSettings settings;
public:
    
    OCVProcessor(OCVProcessorSettings settings, std::vector<OCVProcessorImageModel> models) {
        this->settings = settings;
        this->models = models;
    }
    
    OCVResults processImage(cv::Mat originalImageMat, bool isDebug);
    
private:
    void findItemsAndSubimage(OCVPrivateResult& result, cv::Mat &greyscaleImage);
    void findMistakes(OCVResults& result, OCVPrivateResult& privateResult, std::vector<OCVContour>& ocvContours, bool isDebug);
    void findMistakes(OCVPrivateResult& result, std::vector<OCVContour> ocvContours);
    void debugPreviewResults(OCVResults &result, OCVPrivateResult& privateResult, std::vector<OCVContour> ocvContours);
};

#endif // OCVProcessor_h
#endif // IOS_SIMULATOR

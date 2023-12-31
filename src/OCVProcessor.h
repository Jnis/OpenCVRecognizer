//
//  OCVProcessor.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

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
#import "OCVHungarianAlgorithm.h"

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
    OCVResults processGreyscaleImage(cv::Mat greyscaleImageMat, bool isDebug);
    void findItemsAndRect(OCVPrivateResult& result, cv::Mat &greyscaleImage);
    void prepareHungarianMatrix(VVInt& matrix, OCVItemPrivateResult* itemResult, std::vector<OCVContour>& ocvContours);
    void adjustContoursFit(OCVItemPrivateResult* itemResult, std::vector<OCVContour> &ocvContours, cv::Size ocvContoursMatSize);
    void adjustContours(OCVItemPrivateResult* itemResult, std::vector<OCVContour> &ocvContours, cv::Size ocvContoursMatSize);
    void findAndFilterByMistakes(OCVPrivateResult& privateResult, std::vector<OCVContour>& ocvContours, cv::Size ocvContoursMatSize);
    int findMistakes(OCVItemPrivateResult* itemResult, std::vector<OCVContour> ocvContours, cv::Size ocvContoursMatSize);
    void debugPreviewResults(OCVResults &result, OCVPrivateResult& privateResult, std::vector<OCVContour> ocvContours, cv::Size ocvContoursMatSize);
    void debugPreview(cv::Mat &debugMat, OCVItemPrivateResult* itemResult, std::vector<OCVContour> ocvContours);
};

#endif // OCVProcessor_h

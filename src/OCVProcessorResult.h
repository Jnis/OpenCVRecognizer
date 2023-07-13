//
//  OCVProcessorResult.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef OCVProcessorResult_h
#define OCVProcessorResult_h

#import <opencv2/opencv2.h>

struct OCVItemResult {
    std::string key;
    float matchPercent;
    int mistakes;

    OCVItemResult(std::string key, float matchPercent, int mistakes) : key(key), matchPercent(matchPercent), mistakes(mistakes) {}
};

struct OCVResults {
    std::vector<cv::Mat> debugMats; // it is updated if isDebug = true
    std::vector<OCVItemResult> items;
};

#endif // OCVProcessorResult_h

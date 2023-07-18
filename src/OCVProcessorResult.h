//
//  OCVProcessorResult.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef OCVProcessorResult_h
#define OCVProcessorResult_h

#ifdef __APPLE__
#import <opencv2/opencv2.h>
#elif __ANDROID__
#include <opencv2/core.hpp>
#endif

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

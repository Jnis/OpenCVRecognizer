//
//  OCVContour.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef OCVContour_h
#define OCVContour_h

#ifdef __APPLE__
#import <opencv2/opencv2.h>
#elif __ANDROID__
#include <opencv2/core.hpp>
#endif

struct OCVContour {
    std::vector<cv::Point> contour;
    cv::Point center;
    float area;
};

extern std::vector<OCVContour> convertContours(const std::vector<std::vector<cv::Point>> &contours);
extern void scaleContours(std::vector<OCVContour> &cvContours, float x, float y);
extern void rotateContours(std::vector<OCVContour> &cvContours, float angle);
extern void moveContours(std::vector<OCVContour> &cvContours, float x, float y);
extern float distance2Points(float x1, float y1, float x2, float y2);
extern float angleOfVec2D(float x, float y);

#endif // OCVContour_h

//
//  OCVContour.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef OCVContour_h
#define OCVContour_h

#import <opencv2/opencv2.h>

struct OCVContour {
    std::vector<cv::Point> contour;
    cv::Rect rect;
    cv::Point center;
    double area;
    double arcLength;
};

extern std::vector<OCVContour> convertContours(const std::vector<std::vector<cv::Point>> &contours);
extern std::vector<OCVContour> scaleContours(std::vector<OCVContour> cvContoursOld, double k);

#endif // OCVContour_h

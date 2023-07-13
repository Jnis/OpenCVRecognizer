//
//  OCVContour.m
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#import "OCVContour.h"

std::vector<OCVContour> convertContours(const std::vector<std::vector<cv::Point>> &contours) {
    std::vector<OCVContour> cvContours;
    for(int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        cv::Rect rect = cv::boundingRect(contour);
        cv::Point center = cv::Point(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
        OCVContour cvContour;
        cvContour.contour = contour;
        cvContour.rect = rect;
        cvContour.center = center;
        cvContour.area = cv::contourArea(contour);;
        cvContour.arcLength = cv::arcLength(contour, true);
        cvContours.insert(cvContours.end(), cvContour);
    }
    return cvContours;
}

std::vector<OCVContour> scaleContours(std::vector<OCVContour> cvContoursOld, double k) {
    std::vector<OCVContour> cvContours = cvContoursOld;
    for (int i = 0; i < cvContours.size(); i++) {
        cvContours[i].rect.x *= k;
        cvContours[i].rect.y *= k;
        cvContours[i].rect.width *= k;
        cvContours[i].rect.height *= k;
        cvContours[i].center.x *= k;
        cvContours[i].center.y *= k;
        cvContours[i].area *= k * k;
        cvContours[i].arcLength *= k;
        
        for (int j = 0; j < cvContours[i].contour.size(); j++) {
            cvContours[i].contour[j].x *= k;
            cvContours[i].contour[j].y *= k;
        }
    }
    return cvContours;
}

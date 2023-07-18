//
//  OCVContour.m
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#import "OCVContour.h"

#pragma mark - helpers

float distance2Points(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

void rotateVec2D(float &x, float &y, float sinA, float cosA) {
    float px = x * cosA - y * sinA;
    float py = x * sinA + y * cosA;
    x = px;
    y = py;
}

void rotateVec2D(int &x, int &y, float sinA, float cosA) {
    float px = x * cosA - y * sinA;
    float py = x * sinA + y * cosA;
    x = px;
    y = py;
}

void rotateVec2D(float &x, float &y, float angle) {
    float cs = std::cos(angle);
    float sn = std::sin(angle);
    rotateVec2D(x, y, sn, cs);
}

float angleOfVec2D(float x, float y) {
    return std::atan2(y, x);
}

#pragma mark -

std::vector<OCVContour> convertContours(const std::vector<std::vector<cv::Point>> &contours) {
    std::vector<OCVContour> cvContours;
    for(int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        cv::Rect rect = cv::boundingRect(contour);
        cv::Point center = cv::Point(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
        OCVContour cvContour;
        cvContour.contour = contour;
//        cvContour.rect = rect;
        cvContour.center = center;
        cvContour.area = cv::contourArea(contour);;
//        cvContour.arcLength = cv::arcLength(contour, true);
        cvContours.insert(cvContours.end(), cvContour);
    }
    return cvContours;
}

void scaleContours(std::vector<OCVContour> &cvContours, float x, float y) {
    for (int i = 0; i < cvContours.size(); i++) {
        cvContours[i].center.x *= x;
        cvContours[i].center.y *= x;
        cvContours[i].area *= x * y;
        for (int j = 0; j < cvContours[i].contour.size(); j++) {
            cvContours[i].contour[j].x *= x;
            cvContours[i].contour[j].y *= y;
        }
    }
}

void moveContours(std::vector<OCVContour> &cvContours, float x, float y) {
    for (int i = 0; i < cvContours.size(); i++) {
        cvContours[i].center.x += x;
        cvContours[i].center.y += y;
        for (int j = 0; j < cvContours[i].contour.size(); j++) {
            cvContours[i].contour[j].x += x;
            cvContours[i].contour[j].y += y;
        }
    }
}

void rotateContours(std::vector<OCVContour> &cvContours, float angle) {
    float cs = std::cos(angle);
    float sn = std::sin(angle);
    for (int i = 0; i < cvContours.size(); i++) {
        rotateVec2D(cvContours[i].center.x, cvContours[i].center.y, sn, cs);
        for (int j = 0; j < cvContours[i].contour.size(); j++) {
            rotateVec2D(cvContours[i].contour[j].x, cvContours[i].contour[j].y, sn, cs);
        }
    }
}

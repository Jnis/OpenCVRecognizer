//
//  OCVProcessor.m
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#import "OCVProcessor.h"
#import <opencv2/opencv2.h>
#import "OCVHungarianAlgorithm.h"
#include <iostream>

class OCVItemPrivateResult {
public:
    OCVProcessorImageModel* model;
    CGFloat matchPercent;
    VVInt matrix;
    int mistakes;
    
    OCVItemPrivateResult(OCVProcessorImageModel* model) {
        this->model = model;
        matchPercent = 0;
        mistakes = 0;
    }
};

class OCVPrivateResult {
public:
    std::vector<OCVItemPrivateResult> items;
    cv::Mat imageMat;
};

#pragma mark - helpers

double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

double distance(const cv::Point& point1, const cv::Point& point2) {
    return distance(point1.x, point1.y, point2.x, point2.y);
}

bool compareOCVItemPrivateResult(OCVItemPrivateResult i1, OCVItemPrivateResult i2) {
    return i1.mistakes == i2.mistakes ? (i1.matchPercent > i2.matchPercent) : (i1.mistakes < i2.mistakes);
}

#pragma mark -

void OCVProcessor::findItemsAndSubimage(OCVPrivateResult& result, cv::Mat &greyscaleImageMat) {
    double croppedImageMatThresholdFire = 0;
    
    for (int i = 0; i < models.size(); i++) {
        OCVProcessorImageModel *model = &models[i];
        for (int scaleIndex = 0; scaleIndex < model->imageMats.size(); scaleIndex++) {
            cv::Mat testImageMat = model->imageMats[scaleIndex];
            
            cv::Mat res(greyscaleImageMat.rows - testImageMat.rows + 1, greyscaleImageMat.cols - testImageMat.cols + 1, CV_32FC1);
            cv::matchTemplate(greyscaleImageMat, testImageMat, res, TM_CCOEFF_NORMED);
            cv::threshold(res, res, settings.thresholdMin, 1., THRESH_TOZERO);
            
            double minval, maxval;
            cv::Point minloc, maxloc;
            cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);
            
            if (maxval >= settings.thresholdMin) {
                OCVLog("%s %i, %lf %lf", model->key.c_str(), scaleIndex, minval, maxval);
                
                if (result.items.size() == 0 || result.items[result.items.size() - 1].model != model) {
                    OCVItemPrivateResult item(model);
                    result.items.insert(result.items.end(), item);
                }
                result.items[result.items.size() - 1].matchPercent = MAX(maxval, result.items[result.items.size() - 1].matchPercent);
                
                if (maxval >= settings.thresholdFire && maxval > croppedImageMatThresholdFire) {
                    OCVLog("CROPPED: %s %i", model->key.c_str(), scaleIndex);
                    croppedImageMatThresholdFire = maxval;
                    cv::Rect crop(maxloc.x, maxloc.y, testImageMat.cols, testImageMat.rows);
                    result.imageMat = cv::Mat(greyscaleImageMat, crop);
                }
            }
        }
    }
}

void OCVProcessor::findMistakes(OCVPrivateResult& result, std::vector<OCVContour> ocvContours) {
    for (int resultIndex = 0; resultIndex < result.items.size(); resultIndex++) {
        OCVItemPrivateResult* itemResult = &result.items[resultIndex];
        
        double scaleFactor = (double)itemResult->model->originalImageMat.size().width / (double)result.imageMat.size().width;
        std::vector<OCVContour> ocvContoursFixed = scaleContours(ocvContours, scaleFactor);
        
        const double r = itemResult->model->originalImageMat.size().width * settings.radiusKoeff;
        
        VVInt matrix;
        int n = MAX((int)itemResult->model->ocvContours.size(), (int)ocvContoursFixed.size());
        for (int i = 0; i < n; i++) {
            VInt v;
            for (int j = 0; j < n; j++) {
                BOOL isDetected = false;
                
                if (i < itemResult->model->ocvContours.size() && j < ocvContoursFixed.size()) {
                    if (distance(itemResult->model->ocvContours[i].center, ocvContoursFixed[j].center) < r) {
                        // double matchResult = cv::matchShapes(model.model.cvContours[i].contour, cvContoursFixed[j].contour, CONTOURS_MATCH_I1, 0);
                        double areaDiff = 1 - MIN(ocvContoursFixed[j].area, itemResult->model->ocvContours[i].area) / MAX(ocvContoursFixed[j].area, itemResult->model->ocvContours[i].area);
                        if (areaDiff < settings.maxAreaDifferenceKoeff) {
                            isDetected = true;
                        }
                    }
                }
                
                v.push_back(isDetected ? 0 : 1);
            }
            matrix.insert(matrix.end(), v);
        }
        
        VPInt hungarianResult = hungarian(matrix);
        
        for (int i = 0; i < matrix.size(); i++) {
            NSString *str = @"";
            for (int j = 0; j < matrix[i].size(); j++) {
                str = [str stringByAppendingFormat:@" %i", matrix[i][j]];
            }
            OCVLog("%@", str);
        }
        
        int mistakes = 0;
        for (int i = 0; i < hungarianResult.size(); i++) {
            int second = hungarianResult[i].second;
            int first = hungarianResult[i].first;
            int res = matrix[first][second];
            OCVLog("%i %i = %i", second, first, res);
            mistakes += res;
        }
        
        itemResult->matrix = matrix;
        itemResult->mistakes = mistakes;
        OCVLog("%s mistakes = %i", itemResult->model->key.c_str(), mistakes);
    }
}

void OCVProcessor::findMistakes(OCVResults& result, OCVPrivateResult& privateResult, std::vector<OCVContour>& ocvContours, bool isDebug) {
    // cv::blur(croppedImageMat, croppedImageMat, cv::Size(3, 3));
    cv::threshold(privateResult.imageMat, privateResult.imageMat, 130, 255, THRESH_BINARY_INV);
    // cv::Canny(croppedImageMat, croppedImageMat, 100,200);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(privateResult.imageMat, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    ocvContours = convertContours(contours);
    
    // remove artefacts
    for (int i = (int)ocvContours.size() - 1; i >= 0 ; i--) {
        if (ocvContours[i].area < settings.artefactArea) {
            OCVLog("Contours artefact deleted: area = %.3f", ocvContours[i].area);
            ocvContours.erase(ocvContours.begin() + i);
        }
    }
    
    if (isDebug){
        result.debugMats.insert(result.debugMats.end(), privateResult.imageMat);
    }
    
    findMistakes(privateResult, ocvContours);
    
    // filter results with too many mistakes
    for (int resultIndex = (int)privateResult.items.size() - 1; resultIndex >= 0; resultIndex--) {
        OCVItemPrivateResult* itemResult = &privateResult.items[resultIndex];
        int maxMistakes = itemResult->matrix.size() * 0.5;
        if (itemResult->mistakes > maxMistakes) {
            OCVLog("REMOVED: %s mistakes %i / %i (too many mistakes)", itemResult->model->key.c_str(), itemResult->mistakes, (int)itemResult->matrix.size());
            privateResult.items.erase(privateResult.items.begin() + resultIndex);
        }
    }
}

OCVResults OCVProcessor::processImage(cv::Mat originalImageMat, bool isDebug) {
    OCVResults result;
    float k = settings.minVideoSize / MIN(originalImageMat.cols, originalImageMat.rows);
    
    cv::Mat greyscaleImageMat;
    cv::resize(originalImageMat, greyscaleImageMat, cv::Size(originalImageMat.cols * k, originalImageMat.rows * k), INTER_LINEAR);
    cv::cvtColor(greyscaleImageMat, greyscaleImageMat, COLOR_BGR2GRAY);
    
    // 1. try to find an Image on the photo and Rect for it
    OCVPrivateResult privateResult;
    findItemsAndSubimage(privateResult, greyscaleImageMat);
    
    // 2. compare contours for every Image and cropped photo
    if (privateResult.imageMat.data != NULL) {
        if (isDebug){
            cv::Mat copyMat;
            privateResult.imageMat.copyTo(copyMat);
            result.debugMats.insert(result.debugMats.end(), copyMat);
        }
        
        std::vector<OCVContour> ocvContours;
        if (settings.findMistakes) {
            findMistakes(result, privateResult, ocvContours, isDebug);
        }
        
        std::sort(privateResult.items.begin(), privateResult.items.end(), compareOCVItemPrivateResult);
        
        // remove unnecessary results
        while (privateResult.items.size() > settings.maxResults) {
            privateResult.items.erase(privateResult.items.end() - 1);
        }
        
        if (isDebug){
            if (settings.findMistakes) {
                debugPreviewResults(result, privateResult, ocvContours);
            }
        }
        
        for (int resultIndex = 0; resultIndex < privateResult.items.size(); resultIndex++) {
            OCVItemPrivateResult* itemResult = &privateResult.items[resultIndex];
            result.items.insert(result.items.end(), OCVItemResult(itemResult->model->key, itemResult->matchPercent, itemResult->mistakes));
        }
    }

    return result;
}

void OCVProcessor::debugPreviewResults(OCVResults &result, OCVPrivateResult& privateResult, std::vector<OCVContour> ocvContours) {
    for (int resultIndex = 0; resultIndex < privateResult.items.size(); resultIndex++) {
        OCVItemPrivateResult* itemResult = &privateResult.items[resultIndex];
        cv::Mat debugMat = cv::Mat::zeros(itemResult->model->originalImageMat.size(), CV_8UC3);
        double scaleFactor = (double)itemResult->model->originalImageMat.size().width / (double)privateResult.imageMat.size().width;
        std::vector<OCVContour> ocvContoursFixed = scaleContours(ocvContours, scaleFactor);
        
        for (int ii = 0; ii < itemResult->model->ocvContours.size(); ii++) {
            std::vector<std::vector<cv::Point>> contour;
            contour.insert(contour.begin(), itemResult->model->ocvContours[ii].contour);
            cv::drawContours(debugMat, contour, 0, cv::Scalar(255,255,255) ,1);
            
            cv::drawMarker(debugMat, itemResult->model->ocvContours[ii].center, cv::Scalar(255,0,0), MARKER_DIAMOND, 15);
        }
        for (int ii = 0; ii < ocvContoursFixed.size(); ii++) {
            std::vector<std::vector<cv::Point>> contour;
            contour.insert(contour.begin(), ocvContoursFixed[ii].contour);
            cv::drawContours(debugMat, contour, 0, cv::Scalar(255,255,0) ,1);
            
            cv::drawMarker(debugMat, ocvContoursFixed[ii].center, cv::Scalar(0,255,255), MARKER_TRIANGLE_UP, 10);
        }
        
        OCVLog("%s %i / %.2f:", itemResult->model->key.c_str(), itemResult->mistakes, itemResult->matchPercent);
        
        VPInt hungarianResult = hungarian(itemResult->matrix);
        for (int i = 0; i < hungarianResult.size(); i++) {
            int second = hungarianResult[i].second;
            int first = hungarianResult[i].first;
            int res = itemResult->matrix[first][second];
            OCVLog("%i %i = %i", second, first, res);
            
            if (res == 0) {
                if (first < itemResult->model->ocvContours.size() && second < ocvContoursFixed.size()) {
                    cv::line(debugMat, itemResult->model->ocvContours[first].center, ocvContoursFixed[second].center, cv::Scalar(255,150,0), 2);
                }
            } else {
                const double r = itemResult->model->originalImageMat.size().width * settings.radiusKoeff;
                if (first < itemResult->model->ocvContours.size()) {
                    cv::drawMarker(debugMat, itemResult->model->ocvContours[first].center, cv::Scalar(255,0,0), MARKER_STAR, r, 2);
                }
                if (second < ocvContoursFixed.size()) {
                    cv::drawMarker(debugMat, ocvContoursFixed[second].center, cv::Scalar(255,0,0), MARKER_STAR, r, 2);
                }
            }
        }
        
        result.debugMats.insert(result.debugMats.end(), debugMat);
    }
}

//                std::vector<cv::Point> contour = model.model.contours[i];
//                cv::Rect rect = cv::boundingRect(contour);
//                cv::Point center = cv::Point(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
//                cv::drawMarker(ooo, center, cv::Scalar(255,0,0));
//
//                cv::Vec4f line;
//                cv::fitLine(contour, line, DIST_L2, 0, 0.01, 0.01);
//                float vx = line[0];
//                float vy = line[1];
//
//
//                cv::line(ooo, cv::Point(center.x, center.y),
//                         cv::Point(center.x + vx * 200, center.y + vy * 200), cv::Scalar(0,50,255));
//

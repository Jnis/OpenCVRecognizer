//
//  OCVProcessor.m
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#import "OCVProcessor.h"
#include <iostream>

#ifdef DEBUG
#ifdef __APPLE__
//#import <opencv2/imgcodecs/ios.h>
#elif __ANDROID__
//
#endif
#endif // DEBUG


class OCVItemPrivateResult {
public:
    OCVProcessorImageModel* model;
    float matchPercent;
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

float distance(const cv::Point& point1, const cv::Point& point2) {
    return distance2Points(point1.x, point1.y, point2.x, point2.y);
}

bool compareOCVItemPrivateResult(OCVItemPrivateResult i1, OCVItemPrivateResult i2) {
    return i1.mistakes == i2.mistakes ? (i1.matchPercent > i2.matchPercent) : (i1.mistakes < i2.mistakes);
}

#pragma mark -

void OCVProcessor::findItemsAndSubimage(OCVPrivateResult& result, cv::Mat &greyscaleImageMat) {
    float croppedImageMatThresholdFire = 0;
    
    for (int i = 0; i < models.size(); i++) {
        OCVProcessorImageModel *model = &models[i];
        for (int scaleIndex = 0; scaleIndex < model->imageMats.size(); scaleIndex++) {
            cv::Mat testImageMat = model->imageMats[scaleIndex];
            
            cv::Mat res(greyscaleImageMat.rows - testImageMat.rows + 1, greyscaleImageMat.cols - testImageMat.cols + 1, CV_32FC1);
            cv::matchTemplate(greyscaleImageMat, testImageMat, res, cv::TM_CCOEFF_NORMED);
            cv::threshold(res, res, settings.thresholdMin, 1., cv::THRESH_TOZERO);
            
            double minval, maxval;
            cv::Point minloc, maxloc;
            cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);
            
            if (maxval >= settings.thresholdMin) {
//                OCVLog("%s %i, %lf %lf", model->key.c_str(), scaleIndex, minval, maxval);
                
                if (result.items.size() == 0 || result.items[result.items.size() - 1].model != model) {
                    OCVItemPrivateResult item(model);
                    result.items.insert(result.items.end(), item);
                }
                result.items[result.items.size() - 1].matchPercent = MAX(maxval, result.items[result.items.size() - 1].matchPercent);
                
                if (maxval >= settings.thresholdFire && maxval > croppedImageMatThresholdFire) {
                    OCVLog("DETECTED: %s %i", model->key.c_str(), scaleIndex);
                    croppedImageMatThresholdFire = maxval;
                    cv::Rect crop(maxloc.x, maxloc.y, testImageMat.cols, testImageMat.rows);
                    result.imageMat = cv::Mat(greyscaleImageMat, crop);
                }
            }
        }
    }
}

void OCVProcessor::prepareHungarianMatrix(VVInt& matrix, OCVItemPrivateResult* itemResult, std::vector<OCVContour>& ocvContours) {
    const float r = itemResult->model->originalImageMat.size().width * settings.radiusKoeff;
    int n = MAX((int)itemResult->model->ocvContours.size(), (int)ocvContours.size());
    for (int i = 0; i < n; i++) {
        VInt v;
        for (int j = 0; j < n; j++) {
            bool isDetected = false;
            
            if (i < itemResult->model->ocvContours.size() && j < ocvContours.size()) {
                if (distance(itemResult->model->ocvContours[i].center, ocvContours[j].center) < r) {
                    // float matchResult = cv::matchShapes(model.model.cvContours[i].contour, cvContoursFixed[j].contour, cv::CONTOURS_MATCH_I1, 0);
                    float areaDiff = 1 - MIN(ocvContours[j].area, itemResult->model->ocvContours[i].area) / MAX(ocvContours[j].area, itemResult->model->ocvContours[i].area);
                    if (areaDiff < settings.maxAreaDifferenceKoeff) {
                        isDetected = true;
                    }
                }
            }
            
            v.push_back(isDetected ? 0 : 1);
        }
        matrix.insert(matrix.end(), v);
    }
}

void OCVProcessor::adjustContoursFit(OCVItemPrivateResult* itemResult, std::vector<OCVContour>& ocvContours, cv::Size ocvContoursMatSize) {
    // find paired points
    std::vector<std::pair<int, int>> pairs;
    {
        VVInt matrix;
        prepareHungarianMatrix(matrix, itemResult, ocvContours);
        VPInt hungarianResult = hungarian(matrix);
        for (int i = 0; i < hungarianResult.size(); i++) {
            int first = hungarianResult[i].first;
            int second = hungarianResult[i].second;
            int res = matrix[first][second];
            
            if (res == 0) {
                if (first < itemResult->model->ocvContours.size() && second < ocvContours.size()) {
                    pairs.insert(pairs.end(), std::pair<int, int>(first, second));
                }
            }
        }
    }
    
    if (pairs.size() > 0) {
        { // rotate to correct angle
            float angleSumm = 0;
            int anglesCount = 0;
            for (int i = 0; i < ((int)pairs.size()) - 1; i++) {
                cv::Point firstVector = itemResult->model->ocvContours[pairs[i].first].center - itemResult->model->ocvContours[pairs[i + 1].first].center;
                cv::Point secondVector = ocvContours[pairs[i].second].center - ocvContours[pairs[i + 1].second].center;
                if ((firstVector.x != 0 || firstVector.y != 0) && (secondVector.x != 0 || secondVector.y != 0)) {
                    float firstAngle = angleOfVec2D(firstVector.x, firstVector.y);
                    float secondAngle = angleOfVec2D(secondVector.x, secondVector.y);
                    float angle = firstAngle - secondAngle;
                    if (angle > M_PI) { angle -= M_PI * 2; }
                    if (angle < -M_PI) { angle += M_PI * 2; }
                    angleSumm += angle;
                    anglesCount += 1;
                }
            }
            if (anglesCount > 0) {
                rotateContours(ocvContours, angleSumm / anglesCount);
            }
        }
        
        { // rescale
            cv::Point firstMinPoint = itemResult->model->ocvContours[pairs[0].first].center;
            cv::Point secondMinPoint = ocvContours[pairs[0].second].center;
            cv::Point firstMaxPoint = itemResult->model->ocvContours[pairs[0].first].center;
            cv::Point secondMaxPoint = ocvContours[pairs[0].second].center;
            for (int i = 0; i < pairs.size(); i++) {
                cv::Point firstCenter = itemResult->model->ocvContours[pairs[i].first].center;
                cv::Point secondCenter = ocvContours[pairs[i].second].center;
                firstMinPoint.x = MIN(firstMinPoint.x, firstCenter.x);
                firstMinPoint.y = MIN(firstMinPoint.y, firstCenter.y);
                firstMaxPoint.x = MAX(firstMaxPoint.x, firstCenter.x);
                firstMaxPoint.y = MAX(firstMaxPoint.y, firstCenter.y);
                secondMinPoint.x = MIN(secondMinPoint.x, secondCenter.x);
                secondMinPoint.y = MIN(secondMinPoint.y, secondCenter.y);
                secondMaxPoint.x = MAX(secondMaxPoint.x, secondCenter.x);
                secondMaxPoint.y = MAX(secondMaxPoint.y, secondCenter.y);
            }
            
            moveContours(ocvContours, -secondMinPoint.x, -secondMinPoint.y);
            {
                const float zeroDiff = 10;
                float firstWidth = firstMaxPoint.x - firstMinPoint.x;
                float secondWidth = secondMaxPoint.x - secondMinPoint.x;
                float firstHeight = firstMaxPoint.y - firstMinPoint.y;
                float secondHeight = secondMaxPoint.y - secondMinPoint.y;
                bool canScaleWidth = firstWidth > zeroDiff && secondWidth > zeroDiff;
                bool canScaleHeight = firstHeight > zeroDiff && secondHeight > zeroDiff;
                float xScale = canScaleWidth ? firstWidth / secondWidth : 1;
                float yScale = canScaleHeight ? firstHeight / secondHeight : 1;
                if (canScaleWidth && !canScaleHeight) { yScale = xScale; }
                if (!canScaleWidth && canScaleHeight) { xScale = yScale; }
                scaleContours(ocvContours, xScale, yScale);
            }
            moveContours(ocvContours, firstMinPoint.x, firstMinPoint.y);
        }
    }
}

void OCVProcessor::adjustContours(OCVItemPrivateResult* itemResult, std::vector<OCVContour>& ocvContours, cv::Size ocvContoursMatSize) {
    float scaleFactor = (float)itemResult->model->originalImageMat.size().width / (float)ocvContoursMatSize.width;
    scaleContours(ocvContours, scaleFactor, scaleFactor);
    if (settings.adjustContoursFit) {
        adjustContoursFit(itemResult, ocvContours, ocvContoursMatSize);
    }
}

/// returns max count of mistakes
int OCVProcessor::findMistakes(OCVItemPrivateResult* itemResult, std::vector<OCVContour> ocvContours, cv::Size ocvContoursMatSize) {
    std::vector<OCVContour> ocvContoursFixed = ocvContours;
    adjustContours(itemResult, ocvContoursFixed, ocvContoursMatSize);
    
    VVInt matrix;
    prepareHungarianMatrix(matrix, itemResult, ocvContoursFixed);
    VPInt hungarianResult = hungarian(matrix);
    int mistakes = 0;
    for (int i = 0; i < hungarianResult.size(); i++) {
        int first = hungarianResult[i].first;
        int second = hungarianResult[i].second;
        int res = matrix[first][second];
//        OCVLog("%i %i = %i", second, first, res);
        mistakes += res;
    }
    itemResult->mistakes = mistakes;
//    OCVLog("%s mistakes = %i", itemResult->model->key.c_str(), mistakes);
    return (int)matrix.size();
}

void OCVProcessor::findAndFilterByMistakes(OCVResults& result, OCVPrivateResult& privateResult, std::vector<OCVContour>& ocvContours, bool isDebug) {
    // cv::blur(croppedImageMat, croppedImageMat, cv::Size(3, 3));
    cv::threshold(privateResult.imageMat, privateResult.imageMat, 130, 255, cv::THRESH_BINARY_INV);
    // cv::Canny(croppedImageMat, croppedImageMat, 100,200);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(privateResult.imageMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
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
    
    cv::Size ocvContoursMatSize = cv::Size(privateResult.imageMat.size().width, privateResult.imageMat.size().height);
    for (int resultIndex = (int)privateResult.items.size() - 1; resultIndex >= 0; resultIndex--) {
        OCVItemPrivateResult* itemResult = &privateResult.items[resultIndex];
        int matrixSize = findMistakes(itemResult, ocvContours, ocvContoursMatSize);
        
        // filter results with too many mistakes
        const int maxMistakes = matrixSize * 0.5;
        if (itemResult->mistakes > maxMistakes) {
            // OCVLog("REMOVED: %s mistakes %i / %i (too many mistakes)", itemResult->model->key.c_str(), itemResult->mistakes, (int)itemResult->matrix.size());
            privateResult.items.erase(privateResult.items.begin() + resultIndex);
        }
    }
}

OCVResults OCVProcessor::processImage(cv::Mat originalImageMat, bool isDebug) {
    OCVResults result;
    float k = settings.minVideoSize / MIN(originalImageMat.cols, originalImageMat.rows);
    
    cv::Mat greyscaleImageMat;
    cv::resize(originalImageMat, greyscaleImageMat, cv::Size(originalImageMat.cols * k, originalImageMat.rows * k), cv::INTER_LINEAR);
    cv::cvtColor(greyscaleImageMat, greyscaleImageMat, cv::COLOR_BGR2GRAY);
    
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
            findAndFilterByMistakes(result, privateResult, ocvContours, isDebug);
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

void OCVProcessor::debugPreview(cv::Mat &debugMat, OCVItemPrivateResult* itemResult, std::vector<OCVContour> ocvContours) {
    for (int ii = 0; ii < itemResult->model->ocvContours.size(); ii++) {
        std::vector<std::vector<cv::Point>> contour;
        contour.insert(contour.begin(), itemResult->model->ocvContours[ii].contour);
        cv::drawContours(debugMat, contour, 0, cv::Scalar(255,255,255) ,1);
        
        cv::drawMarker(debugMat, itemResult->model->ocvContours[ii].center, cv::Scalar(255,0,0), cv::MARKER_DIAMOND, 15);
    }
    for (int ii = 0; ii < ocvContours.size(); ii++) {
        std::vector<std::vector<cv::Point>> contour;
        contour.insert(contour.begin(), ocvContours[ii].contour);
        cv::drawContours(debugMat, contour, 0, cv::Scalar(255,255,0) ,1);
        
        cv::drawMarker(debugMat, ocvContours[ii].center, cv::Scalar(0,255,255), cv::MARKER_TRIANGLE_UP, 10);
    }
    
    OCVLog("%s %i / %.2f:", itemResult->model->key.c_str(), itemResult->mistakes, itemResult->matchPercent);
    
    VVInt matrix;
    prepareHungarianMatrix(matrix, itemResult, ocvContours);
    VPInt hungarianResult = hungarian(matrix);
    for (int i = 0; i < hungarianResult.size(); i++) {
        int first = hungarianResult[i].first;
        int second = hungarianResult[i].second;
        int res = matrix[first][second];
//            OCVLog("%i %i = %i", second, first, res);
        
        if (res == 0) {
            if (first < itemResult->model->ocvContours.size() && second < ocvContours.size()) {
                cv::line(debugMat, itemResult->model->ocvContours[first].center, ocvContours[second].center, cv::Scalar(255,150,0), 2);
            }
        } else {
            const float r = itemResult->model->originalImageMat.size().width * settings.radiusKoeff;
            if (first < itemResult->model->ocvContours.size()) {
                cv::drawMarker(debugMat, itemResult->model->ocvContours[first].center, cv::Scalar(255,0,0), cv::MARKER_STAR, r, 2);
            }
            if (second < ocvContours.size()) {
                cv::drawMarker(debugMat, ocvContours[second].center, cv::Scalar(255,0,0), cv::MARKER_STAR, r, 2);
            }
        }
    }
}

void OCVProcessor::debugPreviewResults(OCVResults &result, OCVPrivateResult& privateResult, std::vector<OCVContour> ocvContours) {
    cv::Size ocvContoursMatSize = cv::Size(privateResult.imageMat.size().width, privateResult.imageMat.size().height);
    for (int resultIndex = 0; resultIndex < privateResult.items.size(); resultIndex++) {
        OCVItemPrivateResult* itemResult = &privateResult.items[resultIndex];
        std::vector<OCVContour> ocvContoursFixed = ocvContours;
        adjustContours(itemResult, ocvContoursFixed, ocvContoursMatSize);
        
        cv::Mat debugMat = cv::Mat::zeros(itemResult->model->originalImageMat.size(), CV_8UC3);
        debugPreview(debugMat, itemResult, ocvContoursFixed);
        result.debugMats.insert(result.debugMats.end(), debugMat);
    }
}

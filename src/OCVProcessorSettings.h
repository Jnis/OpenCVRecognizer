//
//  OCVProcessorSettings.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef OCVProcessorSettings_h
#define OCVProcessorSettings_h

struct OCVProcessorSettings {
    // crop original image to detect only this area
    float cropSides = 0.2;
    float cropCenterX = 0.5;
    float cropCenterY = 0.4;
    
    float detectMinVideoSize = 100;
    std::vector<float> detectScales = {0.4, 0.35, 0.3, 0.25};
    std::vector<float> detectAngles = {0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25, -30, 30, -40, 40, -50, 50};
    
    float thresholdDetectFire = 0.8; // % of similarity to start detection
    float thresholdScanFire = 0.6; // % of similarity to start recognition
    float thresholdMin = 0.3; // minimum of % of similarity to additional checking
    
    bool findMistakes = true; // enables find mistakes algorithm: check by polygons
    bool adjustContoursFit = false;
    float artefactAreaK = 0.001; // number of pixels/area to filter contours as artefact
    float radiusKoeff = 0.15; // radius to find polygon
    float maxAreaDifferenceKoeff = 0.6; // area difference to be sure that it is same polygon
    int maxResults = 4;
};

#endif // OCVProcessorSettings_h

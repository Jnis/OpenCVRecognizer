//
//  OCVProcessorSettings.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef IOS_SIMULATOR
#ifndef OCVProcessorSettings_h
#define OCVProcessorSettings_h

struct OCVProcessorSettings {
    float minVideoSize = 320; // rescale photo to this size. Ex. if 320, 800*600 -> 427*320 or 600*800 -> 320*427
    std::vector<float> scales = {0.4, 0.3}; // image size to detect (minVideoSize * sale) TODO: is it possible to use 1 image?
    
    double thresholdFire = 0.7; // % of similarity to start recognition
    double thresholdMin = 0.3; // minimum of % of similarity to additional checking
    
    bool findMistakes = true; // enables find mistakes algorithm: check by polygons
    double artefactArea = 10; // number of pixels/area to filter contours as artefact
    double radiusKoeff = 0.15; // radius to find polygon
    double maxAreaDifferenceKoeff = 0.6; // area difference to be sure that it is same polygon
    int maxResults = 4;
};

#endif // OCVProcessorSettings_h
#endif // IOS_SIMULATOR

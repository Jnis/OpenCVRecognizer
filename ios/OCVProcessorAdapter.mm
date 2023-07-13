//
//  OCVProcessorAdapter.m
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef IOS_SIMULATOR

#import "OCVProcessorAdapter.hpp"
#import "OCVProcessor.h"
#import <opencv2/imgcodecs/ios.h>

@interface UIImage(OCV)
@end
@implementation UIImage(OCV)
+ (UIImage *)debugImage:(UIImage*)debugImage addImage:(UIImage*)uiImage {
    return debugImage == nil ? uiImage : [debugImage debugAddImage:uiImage];
}
- (UIImage *)debugAddImage:(UIImage*)image {
    CGSize newSize = CGSizeMake(MAX(image.size.width, self.size.width), image.size.height + self.size.height);
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 1.0);
    [self drawInRect:CGRectMake(0, 0, self.size.width, self.size.height)];
    [image drawInRect:CGRectMake(0, self.size.height, image.size.width, image.size.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}
@end


@interface OCVProcessorAdapter()
@property(nonatomic, assign) OCVProcessor *processor;
@end

@implementation OCVProcessorAdapter

- (instancetype)initWithModels:(NSArray<OCVAdapterImageModel*>*)models
{
    self = [super init];
    if (self) {
        OCVProcessorSettings settings = OCVProcessorSettings();
        
        std::vector<OCVProcessorImageModel> processorModels;
        for(OCVAdapterImageModel* model in models) {
            cv::Mat imageMat;
            UIImageToMat(model.image, imageMat);
            OCVProcessorImageModel processorModel = OCVProcessorImageModel(model.key.UTF8String, imageMat, settings);
            processorModels.insert(processorModels.end(), processorModel);
        }
        
        self.processor = new OCVProcessor(settings, processorModels);
    }
    return self;
}

- (void) deinit {
    delete self.processor;
}

- (OCVAdapterResults*)processImage:(UIImage*)image {
    cv::Mat imageMat;
    UIImageToMat(image, imageMat);
    OCVResults result = self.processor->processImage(imageMat, true);
    NSMutableArray<OCVAdapterItemResult *> *items = [NSMutableArray array];
    for (int i = 0; i < result.items.size(); i++) {
        [items addObject:[[OCVAdapterItemResult alloc] initWithKey:[NSString stringWithCString:result.items[i].key.c_str()
                                                                                      encoding:[NSString defaultCStringEncoding]]
                                                      matchPercent:result.items[i].matchPercent
                                                          mistakes:result.items[i].mistakes] ];
    }
    UIImage *debugImage = nil;
    for (int i = 0; i < result.debugMats.size(); i++) {
        debugImage = [UIImage debugImage:debugImage addImage:MatToUIImage(result.debugMats[i])];
    }
    return [[OCVAdapterResults alloc] initWithDebugImage:debugImage items:items];
}

@end

#endif // IOS_SIMULATOR

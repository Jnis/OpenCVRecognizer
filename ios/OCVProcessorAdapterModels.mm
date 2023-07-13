//
//  OpencvCameraView.m
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#ifndef IOS_SIMULATOR

#import "OCVProcessorAdapterModels.hpp"

@implementation OCVAdapterImageModel
- (instancetype)initWithImage:(UIImage*)image key:(NSString*)key
{
    self = [super init];
    if (self) {
        self.key = key;
        self.image = image;
    }
    return self;
}
@end


@implementation OCVAdapterItemResult
- (instancetype)initWithKey:(NSString*)key matchPercent:(CGFloat)matchPercent mistakes:(NSInteger)mistakes
{
    self = [super init];
    if (self) {
        self.key = key;
        self.matchPercent = matchPercent;
        self.mistakes = mistakes;
    }
    return self;
}
@end


@implementation OCVAdapterResults
- (instancetype)initWithDebugImage:(UIImage*)debugImage items:(NSArray<OCVAdapterItemResult*>*)items
{
    self = [super init];
    if (self) {
        self.debugImage = debugImage;
        self.items = items;
    }
    return self;
}
@end

#endif // IOS_SIMULATOR

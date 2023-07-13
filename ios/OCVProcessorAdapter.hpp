//
//  OCVProcessorAdapter.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#import <UIKit/UIKit.h>
#import "OCVProcessorAdapterModels.hpp"

NS_ASSUME_NONNULL_BEGIN

@interface OCVProcessorAdapter : NSObject
- (instancetype)initWithModels:(NSArray<OCVAdapterImageModel*>*)models;
- (OCVAdapterResults*)processImage:(UIImage*)image;
@end

NS_ASSUME_NONNULL_END

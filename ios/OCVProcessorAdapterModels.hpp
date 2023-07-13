//
//  OCVProcessor.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface OCVAdapterImageModel : NSObject
@property(nonatomic, strong) NSString *key;
@property(nonatomic, assign) UIImage *image;
- (instancetype)initWithImage:(UIImage*)image key:(NSString*)key;
@end

@interface OCVAdapterItemResult : NSObject
@property(nonatomic, strong) NSString *key;
@property(nonatomic, assign) CGFloat matchPercent;
@property(nonatomic, assign) NSInteger mistakes;
- (instancetype)initWithKey:(NSString*)key matchPercent:(CGFloat)matchPercent mistakes:(NSInteger)mistakes;
@end

@interface OCVAdapterResults : NSObject
@property(nonatomic, strong) UIImage *debugImage;
@property(nonatomic, strong) NSArray<OCVAdapterItemResult*> *items;
- (instancetype)initWithDebugImage:(UIImage*)debugImage items:(NSArray<OCVAdapterItemResult*>*)items;
@end

NS_ASSUME_NONNULL_END

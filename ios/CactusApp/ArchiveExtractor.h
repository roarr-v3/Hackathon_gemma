#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ArchiveExtractor : NSObject

+ (BOOL)extractZipAtURL:(NSURL *)archiveURL
         toDirectoryURL:(NSURL *)directoryURL
                  error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#import "ArchiveExtractor.h"

// libarchive ships with iOS, but its public C headers are not included in the
// iOS SDK. Keep the small ABI surface used by this extractor declared here.
struct archive;
struct archive_entry;

extern struct archive *archive_read_new(void);
extern int archive_read_support_filter_all(struct archive *);
extern int archive_read_support_format_zip(struct archive *);
extern int archive_read_open_filename(struct archive *, const char *, size_t);
extern int archive_read_next_header(struct archive *, struct archive_entry **);
extern int archive_read_extract(struct archive *, struct archive_entry *, int);
extern int archive_read_free(struct archive *);
extern const char *archive_error_string(struct archive *);
extern const char *archive_entry_pathname(struct archive_entry *);
extern void archive_entry_set_pathname(struct archive_entry *, const char *);

#define ARCHIVE_EOF 1
#define ARCHIVE_OK 0
#define ARCHIVE_EXTRACT_PERM 0x0002
#define ARCHIVE_EXTRACT_TIME 0x0004
#define ARCHIVE_EXTRACT_SECURE_SYMLINKS 0x0100
#define ARCHIVE_EXTRACT_SECURE_NODOTDOT 0x0200
#define ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS 0x10000

@implementation ArchiveExtractor

+ (BOOL)extractZipAtURL:(NSURL *)archiveURL
         toDirectoryURL:(NSURL *)directoryURL
                  error:(NSError **)error {
    NSFileManager *fileManager = NSFileManager.defaultManager;
    if (![fileManager createDirectoryAtURL:directoryURL
               withIntermediateDirectories:YES
                                attributes:nil
                                     error:error]) {
        return NO;
    }

    struct archive *reader = archive_read_new();
    archive_read_support_filter_all(reader);
    archive_read_support_format_zip(reader);

    int result = archive_read_open_filename(reader, archiveURL.fileSystemRepresentation, 1024 * 1024);
    if (result != ARCHIVE_OK) {
        [self setError:error fromArchive:reader fallback:@"The model archive could not be opened."];
        archive_read_free(reader);
        return NO;
    }

    struct archive_entry *entry = NULL;
    const int flags = ARCHIVE_EXTRACT_TIME |
                      ARCHIVE_EXTRACT_PERM |
                      ARCHIVE_EXTRACT_SECURE_NODOTDOT |
                      ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS |
                      ARCHIVE_EXTRACT_SECURE_SYMLINKS;

    while ((result = archive_read_next_header(reader, &entry)) == ARCHIVE_OK) {
        const char *rawPath = archive_entry_pathname(entry);
        if (rawPath == NULL) {
            [self setError:error fromArchive:reader fallback:@"The model archive contains an invalid path."];
            archive_read_free(reader);
            return NO;
        }

        NSString *relativePath = [NSString stringWithUTF8String:rawPath];
        if (relativePath.absolutePath || [relativePath.pathComponents containsObject:@".."]) {
            if (error) {
                *error = [NSError errorWithDomain:@"CactusArchive"
                                             code:2
                                         userInfo:@{NSLocalizedDescriptionKey:
                                                        @"The model archive contains an unsafe path."}];
            }
            archive_read_free(reader);
            return NO;
        }

        NSURL *destinationURL = [directoryURL URLByAppendingPathComponent:relativePath];
        archive_entry_set_pathname(entry, destinationURL.fileSystemRepresentation);
        result = archive_read_extract(reader, entry, flags);
        if (result != ARCHIVE_OK) {
            [self setError:error fromArchive:reader fallback:@"The model archive could not be extracted."];
            archive_read_free(reader);
            return NO;
        }
    }

    if (result != ARCHIVE_EOF) {
        [self setError:error fromArchive:reader fallback:@"The model archive ended unexpectedly."];
        archive_read_free(reader);
        return NO;
    }

    archive_read_free(reader);
    return YES;
}

+ (void)setError:(NSError **)error
      fromArchive:(struct archive *)archive
         fallback:(NSString *)fallback {
    if (!error) {
        return;
    }
    const char *message = archive_error_string(archive);
    NSString *description = message ? [NSString stringWithUTF8String:message] : fallback;
    *error = [NSError errorWithDomain:@"CactusArchive"
                                 code:1
                             userInfo:@{NSLocalizedDescriptionKey: description}];
}

@end

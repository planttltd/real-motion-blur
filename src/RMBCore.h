// AE-independent core: optical flow + Metal motion blur on AE-layout buffers.
#pragma once
#import <Metal/Metal.h>
#include <vector>

struct RMBFrame {
    id<MTLBuffer> buf;  // float4 B,G,R,A per pixel, straight alpha (as AE passes it); nil = frame unavailable
    uint32_t pitch;     // in pixels
    uint32_t width, height;
};

// Frames at t-1, t, t+1. src is the layer being blurred (src[1] required); track is what motion is measured on
// (the Motion Src layer, or src again); matte is the FG1 matte layer (all nil when none is set).
struct RMBInputs {
    RMBFrame src[3], track[3], matte[3];
};

enum RMBTrackFrame { RMBTrackAuto = 0, RMBTrackPrevious, RMBTrackNext };
enum RMBDisplay { RMBDisplayBlurred = 0, RMBDisplayUnblurred, RMBDisplayFG, RMBDisplayBG };

// A track point or mask sample: where it is on the current frame (x0, y0) and on the neighbour (x1, y1), in layer
// pixels at the current resolution.
struct RMBTrackPair { float x0, y0, x1, y1; };

// User guidance for one layer's motion. With imageTrack the tracker is pinned to the pairs; without it the motion
// is only interpolated from them (no pairs: no motion).
struct RMBGuides {
    std::vector<RMBTrackPair> prev, next;  // pairs against the previous / next frame
    bool imageTrack = true;
};
enum RMBLayer { RMBLayerBG = 0, RMBLayerFG = 1 };

struct RMBSettings {
    float amount;         // "Blur Amount" (background layer when a matte is set): 1.0 = one frame of motion
    float sensitivity;    // "Motion Sensitivity" / 100
    bool legacy;          // true: Legacy algorithm (pyramid LK flow + gather blur); false: new method
    bool hq;              // AE Best quality: Legacy tracks at 1/2 res and samples bilinear; Draft: 1/4 res, nearest
    int trackFrame = RMBTrackAuto;
    bool trackRGBA = false;     // include alpha in the tracking luma
    bool trackDelinearize = false;  // "Track Enhance: Delinearize": tracking image ^2.2
    int display = RMBDisplayBlurred;
    int matteChannel = 3;       // 0 Red, 1 Green, 2 Blue, 3 Alpha
    float matteShrink = 0;      // "Inv Matte Shrink" in pixels (already scaled for AE downsampling)
    float fgAmount = 0.5f, fgSensitivity = 0.7f;
    RMBGuides guides[2];        // [RMBLayerBG]: the whole frame (no matte) or the background; [RMBLayerFG]: FG1
};

// Blurs inputs.src[1] into dst. Encodes onto queue; blocks until the GPU finishes only if wait is true.
// (Track Frame "Auto" reads back a frame difference first, which waits briefly.)
bool RMBRender(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBInputs& in, const RMBFrame& dst,
               RMBSettings s, bool wait);

// Vector Input: blurs src along a motion-vector layer (red = x, green = y, 0.5 = still, +y up; vectors are valid
// where the vector layer is opaque, the rest is filled in).
struct RMBVectorSettings {
    float amount = 0.5f;
    float maxDisplace = 64;      // pixels a channel value of 0 or 1 stands for
    float scaleX = 1, scaleY = 1;  // already multiplied by the AE downsample factor
    bool hq = true;
    bool linear = false;         // "Vectors are linear": AE linearised the vector layer; re-encode with sRGB first
};
bool RMBRenderVectors(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBFrame& src, const RMBFrame& vectors,
                      const RMBFrame& dst, const RMBVectorSettings& s, bool wait);

// Appends one line (printf-style) to ~/Library/Logs/RealMotionBlur.log: failures AE would only show as a code.
void RMBLog(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Legacy flow only, for tests: returns the Legacy displacement field (rg32Float, tracking resolution) between
// cur and other, anchored on cur: cur(x) ~ other(x - d). Waits for completion.
id<MTLTexture> RMBLegacyFlowForTest(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBFrame& cur,
                                    const RMBFrame& other, RMBSettings s, const RMBGuides* g = nullptr);

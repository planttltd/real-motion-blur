// Self-check for the core: a textured square moving +30 px/frame over a static noise background.
// Build + run via ./build.sh test. Writes build/test_src.png and build/test_blur.png for eyeballing.
#import "../src/RMBCore.h"
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <simd/simd.h>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const uint32_t W = 960, H = 540, SQ = 200;

// Square texture: a low frequency (trackable at 1/4 res) plus a mid frequency that a 30 px streak averages out.
static float Tex(int u, int y) { return 0.6f + 0.12f * sinf(u * 0.07f) * cosf(y * 0.05f) + 0.15f * sinf(u * 0.3f + y * 0.1f); }
// Natural-ish static background. (Pure per-pixel hash noise has no low frequencies, so coarse-to-fine trackers,
// the Legacy one included, leak the square's motion into it at Motion Sensitivity >= 0.5; inherent, not a bug.)
static float Noise(int x, int y) { return 0.4f + 0.15f * sinf(x * 0.05f + 1.3f * sinf(y * 0.031f)) * cosf(y * 0.043f) + 0.05f * sinf(x * 0.21f + y * 0.17f); }

static RMBFrame MakeFrame(id<MTLDevice> dev, int sqx)
{
    RMBFrame f = { [dev newBufferWithLength:W * H * 16 options:MTLResourceStorageModeShared], W, W, H };
    float* p = (float*)f.buf.contents;
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            bool in = (int)x >= sqx && (int)x < sqx + (int)SQ && y >= 170 && y < 170 + SQ;
            float v = in ? Tex(x - sqx, y) : Noise(x, y);
            float* px = p + (y * W + x) * 4;
            px[0] = v; px[1] = v; px[2] = v; px[3] = 1;
        }
    return f;
}

// Matte layer for the square: alpha 1 inside, 0 outside.
static RMBFrame MakeMatte(id<MTLDevice> dev, int sqx)
{
    RMBFrame f = { [dev newBufferWithLength:W * H * 16 options:MTLResourceStorageModeShared], W, W, H };
    float* p = (float*)f.buf.contents;
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++)
            p[(y * W + x) * 4 + 3] = (int)x >= sqx && (int)x < sqx + (int)SQ && y >= 170 && y < 170 + SQ;
    return f;
}

static float Alpha(const RMBFrame& f, int x, int y) { return ((float*)f.buf.contents)[(y * f.pitch + x) * 4 + 3]; }

static RMBInputs In(const RMBFrame t[3])
{
    RMBInputs in = {};
    for (int i = 0; i < 3; i++) in.src[i] = t[i];
    return in;
}

static float At(const RMBFrame& f, int x, int y) { return ((float*)f.buf.contents)[(y * f.pitch + x) * 4]; }

static void WritePNG(const RMBFrame& f, const char* path)
{
    std::vector<uint8_t> px(W * H);
    for (uint32_t i = 0; i < W * H; i++) px[i] = (uint8_t)fminf(255, fmaxf(0, ((float*)f.buf.contents)[i * 4] * 255));
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceGray();
    CGContextRef ctx = CGBitmapContextCreate(px.data(), W, H, 8, W, cs, kCGImageAlphaNone);
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    CGImageDestinationRef d = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)[NSURL fileURLWithPath:@(path)], CFSTR("public.png"), 1, nullptr);
    CGImageDestinationAddImage(d, img, nullptr);
    CGImageDestinationFinalize(d);
    CFRelease(d); CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs);
}

static float RowVariance(const RMBFrame& f, int y, int x0, int x1)
{
    double s = 0, s2 = 0; int n = x1 - x0;
    for (int x = x0; x < x1; x++) { s += At(f, x, y); s2 += At(f, x, y) * At(f, x, y); }
    return float(s2 / n - (s / n) * (s / n));
}

int main()
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    RMBFrame track[3] = { MakeFrame(dev, 350), MakeFrame(dev, 380), MakeFrame(dev, 410) };
    RMBFrame dst = MakeFrame(dev, 0);
    const RMBFrame& src = track[1];
    float vs = RowVariance(src, 270, 420, 540);
    WritePNG(src, "build/test_src.png");

    // Legacy flow, 1/2-res tracking: square moved +30 px since the previous frame -> d = +15 at
    // tracking resolution inside the square, ~0 on the static background.
    {
        id<MTLTexture> f = RMBLegacyFlowForTest(dev, q, track[1], track[0], {1, 0.7f, true, true});
        assert(f && f.width == W / 2 && f.height == H / 2);
        std::vector<float> d(f.width * f.height * 2);
        id<MTLBuffer> rb = [dev newBufferWithLength:d.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromTexture:f sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(f.width, f.height, 1)
                   toBuffer:rb destinationOffset:0 destinationBytesPerRow:f.width * 8 destinationBytesPerImage:d.size() * 4];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        const float* v = (const float*)rb.contents;
        auto D = [&](int x, int y, int c) { return v[(y * f.width + x) * 2 + c]; };
        printf("legacy flow: square (%.2f, %.2f)  background (%.2f, %.2f)\n", D(240, 135, 0), D(240, 135, 1), D(60, 30, 0), D(60, 30, 1));
        assert(fabsf(D(240, 135, 0) - 15) < 1.0f && fabsf(D(240, 135, 1)) < 1.0f);
        assert(fabsf(D(60, 30, 0)) < 0.5f && fabsf(D(60, 30, 1)) < 0.5f);
    }

    for (bool legacy : { true, false }) {
        const char* mode = legacy ? "legacy" : "new";
        // Amount 0 is an exact copy.
        assert(RMBRender(dev, q, In(track), dst, {0.0f, 0.7f, legacy, true}, true));
        for (uint32_t y = 0; y < H; y += 7) for (uint32_t x = 0; x < W; x += 7) assert(At(dst, x, y) == At(src, x, y));

        // Amount 1 (360 degree shutter): square smears horizontally over 30 px, static background stays sharp.
        assert(RMBRender(dev, q, In(track), dst, {1.0f, 1.0f, legacy, true}, true));
        WritePNG(dst, legacy ? "build/test_blur_legacy.png" : "build/test_blur_new.png");
        float vd = RowVariance(dst, 270, 420, 540);
        printf("%s: square row variance src %.5f blur %.5f\n", mode, vs, vd);
        assert(vd < vs * 0.5f);                                          // texture inside square got smeared
        assert(fabsf(At(dst, 100, 60) - At(src, 100, 60)) < 1e-3f);      // far background untouched

        // Negative amount sharpens: moves away from the blur.
        RMBFrame sharp = MakeFrame(dev, 0);
        assert(RMBRender(dev, q, In(track), sharp, {-1.0f, 1.0f, legacy, true}, true));
        assert(RowVariance(sharp, 270, 420, 540) > vs);

        // Missing previous / next frame (layer ends): still blurs from the remaining neighbour.
        RMBFrame none = { nil, 0, 0, 0 };
        RMBFrame head[3] = { none, track[1], track[2] }, tail[3] = { track[0], track[1], none };
        for (auto* t : { head, tail }) {
            assert(RMBRender(dev, q, In(t), dst, {1.0f, 1.0f, legacy, true}, true));
            assert(RowVariance(dst, 270, 420, 540) < vs * 0.5f);
        }

        // Track Enhance (delinearized tracking image) still tracks the square.
        {
            RMBSettings d = { 1.0f, 1.0f, legacy, true };
            d.trackDelinearize = true;
            assert(RMBRender(dev, q, In(track), dst, d, true));
            assert(RowVariance(dst, 270, 420, 540) < vs * 0.5f);
        }

        // Draft quality (1/4-res tracking, nearest sampling) still blurs.
        assert(RMBRender(dev, q, In(track), dst, {1.0f, 1.0f, legacy, false}, true));
        assert(RowVariance(dst, 270, 420, 540) < vs * 0.5f);
    }

    // Layered (FG matte) Legacy: the foreground now streaks over the background past its edges, unlike plain
    // Legacy whose background pixels keep zero motion. Previous-frame tracking, 360-degree shutter.
    {
        RMBInputs in = In(track);
        in.matte[0] = MakeMatte(dev, 350); in.matte[1] = MakeMatte(dev, 380); in.matte[2] = MakeMatte(dev, 410);
        RMBSettings st = { 1.0f, 0.7f, true, true };
        st.trackFrame = RMBTrackPrevious;
        st.fgAmount = 1.0f;
        RMBFrame plain = MakeFrame(dev, 0), layered = MakeFrame(dev, 0), fgOnly = MakeFrame(dev, 0), bgOnly = MakeFrame(dev, 0);
        assert(RMBRender(dev, q, In(track), plain, st, true));
        assert(RMBRender(dev, q, in, layered, st, true));
        WritePNG(layered, "build/test_blur_layered.png");
        float trail = fabsf(At(layered, 372, 270) - At(plain, 372, 270)), lead = fabsf(At(layered, 588, 270) - At(plain, 588, 270));
        printf("layered: FG streak past trailing edge %.3f, leading edge %.3f (vs plain)\n", trail, lead);
        assert(trail > 0.02f && lead > 0.02f);
        assert(fabsf(At(layered, 100, 60) - At(src, 100, 60)) < 1e-3f);  // far background untouched
        assert(fabsf(Alpha(layered, 480, 270) - 1) < 1e-3f && fabsf(Alpha(layered, 100, 60) - 1) < 1e-3f);

        st.display = RMBDisplayFG;
        assert(RMBRender(dev, q, in, fgOnly, st, true));
        assert(Alpha(fgOnly, 100, 60) < 1e-3f && Alpha(fgOnly, 480, 270) > 0.99f);
        st.display = RMBDisplayBG;
        assert(RMBRender(dev, q, in, bgOnly, st, true));
        WritePNG(bgOnly, "build/test_bg_only.png");
        assert(Alpha(bgOnly, 480, 270) > 0.99f);  // hole behind the FG is filled
        // New mode with the same matte (depth + cleaned velocities) also streaks the FG over the background.
        RMBSettings ns = { 1.0f, 0.7f, false, true };
        RMBFrame newMatte = MakeFrame(dev, 0), newPlain = MakeFrame(dev, 0);
        assert(RMBRender(dev, q, In(track), newPlain, ns, true));
        assert(RMBRender(dev, q, in, newMatte, ns, true));
        float nTrail = fabsf(At(newMatte, 372, 270) - At(newPlain, 372, 270));
        printf("new + matte: FG streak past trailing edge %.3f (vs no matte)\n", nTrail);
        assert(nTrail > 0.02f && std::isfinite(At(newMatte, 480, 270)));

        float bgFill = At(bgOnly, 480, 270);
        printf("layered: background filled behind FG with %.3f (square %.3f, background ~0.4)\n", bgFill, At(src, 480, 270));
        assert(bgFill > 0.2f && bgFill < 0.6f);
    }

    // Straight alpha (as AE passes it): an opaque square (colour 0.7) on a fully transparent background whose colour
    // is junk (0.1). The streak's straight colour stays 0.7; blurring straight colour would mix in the junk.
    for (bool legacy : { true, false }) {
        RMBFrame s3[3];
        for (int i = 0; i < 3; i++) {
            s3[i] = MakeMatte(dev, 350 + 30 * i);
            float* p = (float*)s3[i].buf.contents;
            for (uint32_t k = 0; k < W * H; k++) p[k * 4] = p[k * 4 + 1] = p[k * 4 + 2] = p[k * 4 + 3] > 0 ? 0.7f : 0.1f;
        }
        RMBInputs in = In(s3);
        for (int i = 0; i < 3; i++) in.track[i] = track[i];  // track the textured version (Motion Src)
        assert(RMBRender(dev, q, in, dst, {1.0f, 1.0f, legacy, true}, true));
        float a = Alpha(dst, 372, 270), c = At(dst, 372, 270);
        printf("%s straight alpha: streak alpha %.3f colour %.3f (want 0.7)\n", legacy ? "legacy" : "new", a, c);
        assert(a > 0.05f && a < 0.95f && fabsf(c - 0.7f) < 1e-3f);
    }

    // Guides. Image Track off: the motion comes from the track points alone, spread over the frame.
    {
        RMBGuides g;
        g.imageTrack = false;
        g.prev = { { 480, 270, 450, 270 } };  // a point on the square: moved +30 since the previous frame
        id<MTLTexture> f = RMBLegacyFlowForTest(dev, q, track[1], track[0], {1, 0.7f, true, true}, &g);
        std::vector<float> d(f.width * f.height * 2);
        id<MTLBuffer> rb = [dev newBufferWithLength:d.size() * 4 options:MTLResourceStorageModeShared];
        auto read = [&](id<MTLTexture> t, int x, int y) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
            [bl copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x, y, 0) sourceSize:MTLSizeMake(1, 1, 1)
                       toBuffer:rb destinationOffset:0 destinationBytesPerRow:8 destinationBytesPerImage:8];
            [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return simd_make_float2(((float*)rb.contents)[0], ((float*)rb.contents)[1]);
        };
        // (Normalised convolution fading out with distance: motion near the guides, none far from them.)
        simd_float2 at = read(f, 240, 135), near = read(f, 255, 135), farAway = read(f, 60, 30);
        printf("guides, Image Track off: at the point %.2f, 15 px away %.2f, far away %.2f\n", at.x, near.x, farAway.x);
        assert(fabsf(at.x - 15) < 0.1f && near.x > 5 && near.x <= 15.01f && fabsf(near.y) < 0.01f && fabsf(farAway.x) < 0.1f);
        g.prev.clear();
        f = RMBLegacyFlowForTest(dev, q, track[1], track[0], {1, 0.7f, true, true}, &g);
        assert(read(f, 0, 0).x == 0);  // no points: no motion

        // Image Track on: a point on the static background that claims +30 px pins the flow there; the tracked
        // square is unaffected.
        g.imageTrack = true;
        g.prev = { { 100, 60, 70, 60 } };
        f = RMBLegacyFlowForTest(dev, q, track[1], track[0], {1, 0.7f, true, true}, &g);
        simd_float2 pin = read(f, 50, 30), sq = read(f, 240, 135);
        printf("guides, Image Track on: pinned (%.2f, %.2f), square (%.2f, %.2f)\n", pin.x, pin.y, sq.x, sq.y);
        assert(fabsf(pin.x - 30 * 479.f / 959) < 1e-4f && fabsf(sq.x - 15) < 1.0f);  // 30 px at 1/2 res, corner-aligned

        // Through RMBRender: the guide drags the static background around the point into a horizontal streak.
        RMBSettings st = { 1.0f, 0.7f, true, true };
        st.trackFrame = RMBTrackPrevious;
        st.guides[RMBLayerBG] = g;
        RMBFrame plain = MakeFrame(dev, 0);
        assert(RMBRender(dev, q, In(track), plain, { 1.0f, 0.7f, true, true }, true));
        assert(RMBRender(dev, q, In(track), dst, st, true));
        printf("guides through render: background at the point %.4f -> %.4f\n", At(plain, 100, 60), At(dst, 100, 60));
        assert(fabsf(At(dst, 100, 60) - At(plain, 100, 60)) > 1e-3f);
    }

    // Vector Input: a vector layer saying +30 px in x everywhere blurs the static textured frame horizontally; the
    // vector layer's transparent left half is filled from the right.
    {
        RMBFrame vec = MakeFrame(dev, 0);
        float* p = (float*)vec.buf.contents;
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) {
                float* v = p + (y * W + x) * 4;
                v[0] = 0; v[1] = 0.5f; v[2] = 0.5f + 30.0f / (2 * 64); v[3] = x < W / 2 ? 0 : 1;  // B, G, R, A
            }
        RMBVectorSettings vs2;
        vs2.amount = 1;
        RMBFrame stat = MakeFrame(dev, -1000), out = MakeFrame(dev, 0);  // square off-frame: background only
        float v0 = RowVariance(stat, 270, 100, 800);
        assert(RMBRenderVectors(dev, q, stat, vec, out, vs2, true));
        float vl = RowVariance(out, 270, 100, 400), vr = RowVariance(out, 270, 500, 800);
        printf("vectors: row variance %.5f -> left %.5f right %.5f\n", v0, vl, vr);
        assert(vl < v0 * 0.7f && vr < v0 * 0.7f);
        RMBFrame none = { nil, 0, 0, 0 };
        assert(RMBRenderVectors(dev, q, stat, none, out, vs2, true) && At(out, 300, 300) == At(stat, 300, 300));

        // "Vectors are linear": the same layer linearised (sRGB decode), with the option on, blurs the same.
        RMBFrame lin = MakeFrame(dev, 0), outLin = MakeFrame(dev, 0);
        auto decode = [](float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); };
        for (uint32_t i = 0; i < W * H * 4; i++)
            ((float*)lin.buf.contents)[i] = i % 4 == 3 ? p[i] : decode(p[i]);
        vs2.linear = true;
        assert(RMBRenderVectors(dev, q, stat, lin, outLin, vs2, true));
        vs2.linear = false;
        assert(RMBRenderVectors(dev, q, stat, vec, out, vs2, true));
        float worst = 0;
        for (uint32_t x = 0; x < W; x += 3) worst = fmaxf(worst, fabsf(At(outLin, x, 270) - At(out, x, 270)));
        printf("vectors are linear: max difference vs encoded %.2g\n", worst);
        assert(worst < 1e-3f);
    }

    // A layer over transparency whose straight colour under alpha 0 is junk (AE leaves it arbitrary), whipping
    // 150 px/frame, Legacy with Track RGB+A. The junk used to look like texture to the tracker and punch holes into
    // the layer. Blur Amount 0 must hand the input back bit for bit, junk included.
    {
        auto whip = [&](int t) {
            RMBFrame f = MakeFrame(dev, 0);
            float* p = (float*)f.buf.contents;
            srand(7 + t);
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++) {
                    float* px = p + (y * W + x) * 4;
                    int u = int(x) - (150 + 150 * t);
                    if (u < 0 || u >= 400 || y < 120 || y >= 420) {
                        px[0] = rand() * 4.0f / RAND_MAX; px[1] = rand() * 1.0f / RAND_MAX; px[2] = px[0]; px[3] = 0;
                    } else {  // a screen graphic: coloured bars on white
                        float bar = (y / 12) % 3 == 0 ? 0.25f + 0.6f * float(u % 108) / 108 : 0.95f;
                        px[0] = bar; px[1] = 0.9f * bar; px[2] = 0.97f; px[3] = 1;
                    }
                }
            return f;
        };
        RMBFrame w3[3] = { whip(0), whip(1), whip(2) };
        RMBSettings st = { 0.0f, 0.7f, true, true };
        st.trackRGBA = true;
        assert(RMBRender(dev, q, In(w3), dst, st, true));
        assert(memcmp(dst.buf.contents, w3[1].buf.contents, W * H * 16) == 0);
        st.amount = 0.5f;
        assert(RMBRender(dev, q, In(w3), dst, st, true));
        float minA = 1;
        for (uint32_t y = 124; y < 416; y++)
            for (uint32_t x = 300 + 45; x < 700 - 45; x++) minA = fminf(minA, Alpha(dst, x, y));  // past the streaks
        printf("junk under transparency: layer interior min alpha %.3f\n", minA);
        assert(minA > 0.99f);
    }

    printf("all checks passed\n");
    return 0;
}

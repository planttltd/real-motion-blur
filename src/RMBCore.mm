#import "RMBCore.h"
#include <dlfcn.h>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <simd/simd.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Frame { uint32_t pitch, width, height; };
struct LegacyBlur { float ratio[2], scale[2], mix; int32_t hq; };

struct Ctx {
    std::map<std::string, id<MTLComputePipelineState>> pso;
    id<MTLComputePipelineState> operator[](const char* n) const { return pso.at(n); }
};

Ctx* ContextFor(id<MTLDevice> device)
{
    static std::mutex m;
    static std::map<void*, Ctx> cache;  // one entry per GPU, lives for the process
    std::lock_guard<std::mutex> lock(m);
    auto it = cache.find((__bridge void*)device);
    if (it != cache.end()) return &it->second;

    NSURL* url = nil;
    Dl_info info;
    if (dladdr((void*)&ContextFor, &info)) {  // metallib sits in our bundle's Resources (or next to a test binary)
        NSString* dir = [NSString stringWithUTF8String:info.dli_fname].stringByDeletingLastPathComponent;
        for (NSString* p in @[ [dir stringByAppendingPathComponent:@"../Resources/RMBKernels.metallib"],
                               [dir stringByAppendingPathComponent:@"RMBKernels.metallib"] ])
            if ([NSFileManager.defaultManager fileExistsAtPath:p]) { url = [NSURL fileURLWithPath:p]; break; }
    }
    NSError* err = nil;
    id<MTLLibrary> lib = url ? [device newLibraryWithURL:url error:&err] : nil;
    if (!lib) { RMBLog("cannot load metallib: %s", err.localizedDescription.UTF8String); return nullptr; }
    Ctx c;
    for (NSString* name in lib.functionNames) {
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
        if (!p) { RMBLog("pipeline %s: %s", name.UTF8String, err.localizedDescription.UTF8String); return nullptr; }
        c.pso[name.UTF8String] = p;
    }
    return &(cache[(__bridge void*)device] = c);
}

void Dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger w, NSUInteger h)
{
    [enc setComputePipelineState:p];
    NSUInteger tw = p.threadExecutionWidth, th = std::max<NSUInteger>(1, p.maxTotalThreadsPerThreadgroup / tw);
    if (h == 1) { tw = std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup, 256); th = 1; }
    [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

// GPU scratch memory. Allocating (and the driver mapping) hundreds of fresh full-res textures per frame cost more
// than the compute, so scratch is pooled across renders, and reused within a render once a stage is done with it.
// Resources taken while encoding a render are leased to it and go back to the pool when its command buffer
// completes, so concurrent (MFR) renders never share one.
using TexKey = std::tuple<void*, NSUInteger, NSUInteger, MTLPixelFormat>;
using BufKey = std::pair<void*, NSUInteger>;
struct FreeList {
    std::multimap<TexKey, id<MTLTexture>> tex;
    std::multimap<BufKey, id<MTLBuffer>> buf;
};
struct Pool {
    std::mutex m;
    FreeList free;
    size_t bytes = 0;
} gPool;

// Note: idle pool cap = a quarter of the GPU's recommended working set (~3 GB on a 16 GB Mac)
size_t PoolCap(id<MTLDevice> dev) { return std::max<size_t>(size_t(1) << 30, dev.recommendedMaxWorkingSetSize / 4); }
size_t Bytes(id<MTLResource> r) { return r.allocatedSize; }

struct Lease {
    std::vector<id<MTLResource>> all;     // everything this render holds (returned to the pool on completion)
    std::vector<id<MTLResource>> active;  // in use by the current scopes
    FreeList free;                        // finished with earlier in this render; reusable by later stages
};
thread_local Lease* tLease = nullptr;     // the render being encoded on this thread

template <typename K, typename R> R Take(std::multimap<K, R>& m, const K& k)
{
    auto it = m.find(k);
    if (it == m.end()) return nil;
    R r = it->second;
    m.erase(it);
    return r;
}

id<MTLTexture> Tex(id<MTLDevice> dev, NSUInteger w, NSUInteger h, MTLPixelFormat f)
{
    TexKey k = { (__bridge void*)dev, w, h, f };
    id<MTLTexture> t = nil;
    if (tLease) {
        t = Take(tLease->free.tex, k);
        if (!t) {
            std::lock_guard<std::mutex> lock(gPool.m);
            if ((t = Take(gPool.free.tex, k))) gPool.bytes -= Bytes(t);
        }
    }
    if (!t) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = MTLStorageModePrivate;
        t = [dev newTextureWithDescriptor:d];
        if (tLease) tLease->all.push_back(t);
    } else if (std::find(tLease->all.begin(), tLease->all.end(), t) == tLease->all.end()) {
        tLease->all.push_back(t);
    }
    if (tLease) tLease->active.push_back(t);
    return t;
}

id<MTLBuffer> Buf(id<MTLDevice> dev, NSUInteger len)  // private scratch buffer, pooled like Tex
{
    BufKey k = { (__bridge void*)dev, len };
    id<MTLBuffer> b = nil;
    if (tLease) {
        b = Take(tLease->free.buf, k);
        if (!b) {
            std::lock_guard<std::mutex> lock(gPool.m);
            if ((b = Take(gPool.free.buf, k))) gPool.bytes -= Bytes(b);
        }
    }
    if (!b) {
        b = [dev newBufferWithLength:len options:MTLResourceStorageModePrivate];
        if (tLease) tLease->all.push_back(b);
    } else if (std::find(tLease->all.begin(), tLease->all.end(), b) == tLease->all.end()) {
        tLease->all.push_back(b);
    }
    if (tLease) tLease->active.push_back(b);
    return b;
}

void Release(FreeList& fl, id<MTLResource> r)
{
    if ([r conformsToProtocol:@protocol(MTLTexture)]) {
        id<MTLTexture> t = (id<MTLTexture>)r;
        fl.tex.insert({ { (__bridge void*)t.device, t.width, t.height, t.pixelFormat }, t });
    } else {
        id<MTLBuffer> b = (id<MTLBuffer>)r;
        fl.buf.insert({ { (__bridge void*)b.device, b.length }, b });
    }
}

// Scratch taken inside a Scope, except what it is told to keep, becomes reusable by later stages of the same render
// when the scope ends. Later stages are encoded after it, so the GPU runs them after its last use.
struct Scope {
    size_t mark = tLease ? tLease->active.size() : 0;
    std::vector<id<MTLResource>> keep;
    Scope& operator<<(id<MTLResource> r) { keep.push_back(r); return *this; }
    ~Scope()
    {
        if (!tLease) return;
        std::vector<id<MTLResource>> survivors;
        for (size_t i = mark; i < tLease->active.size(); i++) {
            id<MTLResource> r = tLease->active[i];
            if (std::find(keep.begin(), keep.end(), r) != keep.end()) survivors.push_back(r);
            else Release(tLease->free, r);
        }
        tLease->active.resize(mark);
        tLease->active.insert(tLease->active.end(), survivors.begin(), survivors.end());
    }
};

void ReturnToPool(Lease* lease)
{
    std::lock_guard<std::mutex> lock(gPool.m);
    for (id<MTLResource> r : lease->all) {
        if (gPool.bytes + Bytes(r) > PoolCap(r.device)) continue;  // over the cap: let it be freed
        gPool.bytes += Bytes(r);
        Release(gPool.free, r);
    }
    delete lease;
}

// ---------------------------------------------------------------------------------------------------------------
// Legacy: pyramid Lucas-Kanade flow with block-match propagation, encoded entirely on the GPU.

std::vector<float> GaussTaps(int r)  // w(i) = 2^(-i^2 / (r/3)^2)
{
    std::vector<float> w;
    for (int i = -r; i <= r; i++) w.push_back(std::pow(2.0f, -(i * i) / ((r / 3.0f) * (r / 3.0f))));
    return w;
}
const std::vector<float> kBinomial = { 1 / 16.f, 4 / 16.f, 6 / 16.f, 4 / 16.f, 1 / 16.f };
const std::vector<float> kGauss7 = GaussTaps(7), kGauss5 = GaussTaps(5);
const float kNewDispEpsilon = 40.0f;  // the 2x2 solve skips |det A| <= this
// Largest LK update per iteration, in pixels of the level being solved. A near-singular solve (flat area next to a
// strong edge, e.g. a white page beside transparency) otherwise flings vectors hundreds of pixels and the blur shows
// a hard-edged hole. 3 px keeps the fastest real motion (it comes down the pyramid and from the block match) intact.
const float kMaxStep = 3.0f;

using Pyr = std::vector<id<MTLTexture>>;

#ifdef RMB_PROFILE
// Profiling build only: each stage() starts a new compute pass whose GPU start/end timestamps are sampled.
struct Prof {
    id<MTLCounterSampleBuffer> buf;
    std::vector<std::string> names;
};
Prof* gProf = nullptr;
#endif

struct LegacyEncoder {
    id<MTLDevice> dev;
    Ctx& ctx;
    id<MTLComputeCommandEncoder> enc;  // current encoder; the solver swaps it, so always use le.enc
    id<MTLCommandBuffer> cb;
    id<MTLTexture> onesTex = nil;      // 1x1 coverage of 1: plain (unmatted) tracking

    void setValue(id<MTLTexture> t, float v)
    {
        [enc setTexture:t atIndex:0];
        [enc setBytes:&v length:sizeof v atIndex:0];
        Dispatch(enc, ctx["layerSetValue"], t.width, t.height);
    }
    // Profiling build: begin a new timed pass named `name` (no-op otherwise).
    void stage(const char* name, bool concurrent = false)
    {
#ifdef RMB_PROFILE
        if (!gProf) return;
        [enc endEncoding];
        MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
        d.dispatchType = concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
        NSUInteger i = gProf->names.size();
        d.sampleBufferAttachments[0].sampleBuffer = gProf->buf;
        d.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
        d.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
        gProf->names.push_back(name);
        enc = [cb computeCommandEncoderWithDescriptor:d];
#else
        (void)name; (void)concurrent;
#endif
    }

    // Create before binding any other kernel's arguments: creating it dispatches a kernel of its own.
    id<MTLTexture> ones()
    {
        if (!onesTex) setValue(onesTex = Tex(dev, 1, 1, MTLPixelFormatR32Float), 1);
        return onesTex;
    }
    // Unary texture kernel (texture 0 in, texture 1 out) into a new texture of the input's size and format.
    id<MTLTexture> unary(const char* kernel, id<MTLTexture> in, MTLPixelFormat f = MTLPixelFormatInvalid)
    {
        id<MTLTexture> out = Tex(dev, in.width, in.height, f == MTLPixelFormatInvalid ? in.pixelFormat : f);
        [enc setTexture:in atIndex:0];
        [enc setTexture:out atIndex:1];
        Dispatch(enc, ctx[kernel], in.width, in.height);
        return out;
    }

    void filter(id<MTLTexture> in, id<MTLTexture> out, const std::vector<float>& w)  // in must not alias out
    {
        int r = int(w.size() / 2);
        [enc setTexture:in atIndex:0];
        [enc setTexture:out atIndex:1];
        [enc setBytes:w.data() length:w.size() * sizeof(float) atIndex:0];
        [enc setBytes:&r length:sizeof r atIndex:1];
        [enc setComputePipelineState:ctx["legacyFilter2D"]];
        [enc dispatchThreadgroups:MTLSizeMake((in.width + 15) / 16, (in.height + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    id<MTLTexture> filtered(id<MTLTexture> in, const std::vector<float>& w)
    {
        id<MTLTexture> out = Tex(dev, in.width, in.height, in.pixelFormat);
        filter(in, out, w);
        return out;
    }
    void resample(id<MTLTexture> in, id<MTLTexture> out, simd_float4 scale)
    {
        simd_float2 is = { (in.width - 1.0f) / std::max<float>(1, out.width - 1.0f),
                           (in.height - 1.0f) / std::max<float>(1, out.height - 1.0f) };
        [enc setTexture:in atIndex:0];
        [enc setTexture:out atIndex:1];
        [enc setBytes:&is length:sizeof is atIndex:0];
        [enc setBytes:&scale length:sizeof scale atIndex:1];
        Dispatch(enc, ctx["legacyResample"], out.width, out.height);
    }
    id<MTLTexture> halve(id<MTLTexture> in)  // binomial filter + corner-aligned resample
    {
        Scope sc;
        id<MTLTexture> f = Tex(dev, in.width, in.height, in.pixelFormat);
        filter(in, f, kBinomial);
        id<MTLTexture> out = Tex(dev, std::max<NSUInteger>(1, in.width / 2), std::max<NSUInteger>(1, in.height / 2), in.pixelFormat);
        resample(f, out, simd_make_float4(1, 1, 1, 1));
        sc << out;
        return out;
    }
    // Tracking luma pyramid; with a layer matte the frame is first multiplied by it.
    bool delinearize = false;  // Track Enhance, set per render

    Pyr pyramid(const RMBFrame& fr, int baseHalvings, int levels, bool addAlpha, id<MTLTexture> matte = nil)
    {
        id<MTLTexture> img = Tex(dev, fr.width, fr.height, MTLPixelFormatR32Float), one = ones();
        Frame f = { fr.pitch, fr.width, fr.height };
        int a = addAlpha, m = matte != nil;
        [enc setBuffer:fr.buf offset:0 atIndex:0];
        [enc setBytes:&f length:sizeof f atIndex:1];
        [enc setBytes:&a length:sizeof a atIndex:2];
        [enc setBytes:&m length:sizeof m atIndex:3];
        [enc setTexture:img atIndex:0];
        [enc setTexture:matte ?: one atIndex:1];
        Dispatch(enc, ctx["legacyLuma"], fr.width, fr.height);
        for (int i = 0; i < baseHalvings; i++) {  // the full-res and intermediate images are done after halving
            Scope sc;
            img = halve(img);
            sc << img;
        }
        if (delinearize) img = unary("legacyDelinearize", img);
        Pyr p = { img };
        for (int l = 1; l < levels; l++) p.push_back(halve(p.back()));
        return p;
    }
    // New mode's pyramid: the first halving straight from the frame (newLumaHalf), then as the Legacy pyramid.
    // Same level sizes as pyramid().
    Pyr pyramidNew(const RMBFrame& fr, int baseHalvings, int levels, bool addAlpha)
    {
        id<MTLTexture> img = Tex(dev, std::max<NSUInteger>(1, fr.width / 2), std::max<NSUInteger>(1, fr.height / 2),
                                 MTLPixelFormatR32Float);
        Frame f = { fr.pitch, fr.width, fr.height };
        int a = addAlpha;
        [enc setBuffer:fr.buf offset:0 atIndex:0];
        [enc setBytes:&f length:sizeof f atIndex:1];
        [enc setBytes:&a length:sizeof a atIndex:2];
        [enc setTexture:img atIndex:0];
        Dispatch(enc, ctx["newLumaHalf"], img.width, img.height);
        for (int i = 1; i < baseHalvings; i++) {
            Scope sc;
            img = halve(img);
            sc << img;
        }
        if (delinearize) img = unary("legacyDelinearize", img);
        Pyr p = { img };
        for (int l = 1; l < levels; l++) p.push_back(halve(p.back()));
        return p;
    }
    // Coverage follows the luma's reduction path and is binarised at every step (which erodes it).
    Pyr coverage(id<MTLTexture> cov, int baseHalvings, int levels)
    {
        id<MTLTexture> c = unary("layerBinarize", cov);
        for (int i = 0; i < baseHalvings; i++) {
            Scope sc;
            c = unary("layerBinarize", halve(c));
            sc << c;
        }
        Pyr p = { c };
        for (int l = 1; l < levels; l++) p.push_back(unary("layerBinarize", halve(p.back())));
        return p;
    }
    // Tracking setup shared by both directions: base halvings, pyramid depth and the starting (coarsest) level.
    struct Plan { int halvings, levels, L0; };
    static Plan plan(const RMBFrame& cur, bool hq, float sensitivity)
    {
        Plan p = { hq ? 1 : 2, 1, 0 };
        uint32_t w = cur.width >> p.halvings, h = cur.height >> p.halvings;
        for (int a = w, b = h; (a /= 2) > 2 && (b /= 2) > 2 && !(a == 2 && b == 2);) p.levels++;  // halve while > 2
        // Motion Sensitivity picks the coarsest level: floor(levels + log2(s^2)), clamped.
        float lf = std::log2(sensitivity * sensitivity) + p.levels;
        p.L0 = lf >= 0 ? (lf < p.levels ? int(std::floor(lf)) : p.levels - 1) : 0;
        return p;
    }

    // Returns d anchored on cur: cur(x) ~ other(x - d), at tracking resolution. g: track pairs against other.
    id<MTLTexture> flow(const RMBFrame& cur, const RMBFrame& other, const RMBSettings& s,
                        const std::vector<RMBTrackPair>* g = nullptr, bool imageTrack = true)
    {
        Plan pl = plan(cur, s.hq, s.sensitivity);
        id<MTLTexture> g0 = g ? guideImage(*g, cur, pl) : nil;
        if (!imageTrack) return morph(g0, pl) ?: zeroFlow();
        Pyr I1 = pyramid(cur, pl.halvings, pl.levels, s.trackRGBA), I0 = pyramid(other, pl.halvings, pl.levels, s.trackRGBA);
        Pyr G = g0 ? guidePyramid(g0, I1, pl.L0) : Pyr();
        return solveMany({ { &I1, &I0, pl.L0, nullptr, nullptr, g0 ? &G : nullptr } })[0];
    }

    id<MTLTexture> zeroFlow()
    {
        id<MTLTexture> z = Tex(dev, 1, 1, MTLPixelFormatRG32Float);
        [enc setTexture:z atIndex:0];
        Dispatch(enc, ctx["legacyClear"], 1, 1);
        return z;
    }

    // ---- Guides: track points and mask samples ----

    // Level-0 guide image for a tracking plan: (d.x, d.y, 1, 0) at each guided pixel, where d = p_t - p_n in
    // tracking pixels (the flow's sign: cur(x) ~ other(x - d)). Pairs landing on one pixel are averaged.
    // nil when no pair lands in the frame.
    id<MTLTexture> guideImage(const std::vector<RMBTrackPair>& g, const RMBFrame& full, const Plan& pl)
    {
        if (g.empty()) return nil;
        int W = full.width, H = full.height;
        for (int i = 0; i < pl.halvings; i++) { W = std::max(1, W / 2); H = std::max(1, H / 2); }
        float sx = (W - 1.0f) / std::max(1.0f, full.width - 1.0f), sy = (H - 1.0f) / std::max(1.0f, full.height - 1.0f);
        std::map<std::pair<int, int>, simd_float3> acc;
        for (const RMBTrackPair& p : g) {
            int x = int(sx * p.x0 + 0.5f), y = int(sy * p.y0 + 0.5f);  // truncates toward zero
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            acc[{ x, y }] += simd_make_float3(sx * (p.x0 - p.x1), sy * (p.y0 - p.y1), 1);
        }
        if (acc.empty()) return nil;
        std::vector<simd_float4> e;
        for (auto& [xy, a] : acc) e.push_back(simd_make_float4(xy.first, xy.second, a.x / a.z, a.y / a.z));
        id<MTLTexture> t = Tex(dev, W, H, MTLPixelFormatRGBA32Float);
        setValue(t, 0);
        uint32_t n = uint32_t(e.size());
        [enc setBuffer:[dev newBufferWithBytes:e.data() length:e.size() * sizeof e[0] options:MTLResourceStorageModeShared]
                offset:0 atIndex:0];
        [enc setBytes:&n length:sizeof n atIndex:1];
        [enc setTexture:t atIndex:0];
        Dispatch(enc, ctx["guideScatter"], n, 1);
        return t;
    }
    // Guide image reduced like the luma pyramid `like` (binomial + corner resample, displacements scaled by the
    // size ratio) up to level L0. The weight is not renormalised, so coarse levels hold weighted means.
    Pyr guidePyramid(id<MTLTexture> g0, const Pyr& like, int L0)
    {
        Pyr p = { g0 };
        for (int l = 1; l <= L0; l++) {
            Scope sc;
            id<MTLTexture> f = filtered(p.back(), kBinomial);
            id<MTLTexture> out = Tex(dev, like[l].width, like[l].height, MTLPixelFormatRGBA32Float);
            resample(f, out, simd_make_float4((out.width - 1.0f) / std::max(1.0f, f.width - 1.0f),
                                              (out.height - 1.0f) / std::max(1.0f, f.height - 1.0f), 1, 1));
            sc << out;
            p.push_back(out);
        }
        return p;
    }
    void guideAdjust(id<MTLTexture> in, id<MTLTexture> g, float thr, id<MTLTexture> out)
    {
        [enc setTexture:in atIndex:0];
        [enc setTexture:g atIndex:1];
        [enc setTexture:out atIndex:2];
        [enc setBytes:&thr length:sizeof thr atIndex:0];
        Dispatch(enc, ctx["guideAdjust"], out.width, out.height);
    }
    // Image Track off: the field is interpolated from the guides alone, by normalised Gaussian convolution from
    // coarse to fine (radius 2^l from the sensitivity's start level down), at tracking level 0. nil: no guides.
    id<MTLTexture> morph(id<MTLTexture> g0, const Plan& pl)
    {
        if (!g0) return nil;
        Scope sc;
        auto adjust = [&](id<MTLTexture> v, id<MTLTexture> g, float thr) {
            id<MTLTexture> out = Tex(dev, g0.width, g0.height, MTLPixelFormatRG32Float);
            guideAdjust(v, g, thr, out);
            return out;
        };
        id<MTLTexture> V = Tex(dev, g0.width, g0.height, MTLPixelFormatRG32Float);
        setValue(V, 0);
        for (int l = pl.L0; l >= 1; l--) {
            float R = std::ldexp(1.0f, l), sum = 0;
            int r = int(R);
            std::vector<float> w;  // the blur's normalised 1-D weights; threshold: the squared weight at ~0.625 R
            for (int k = 0; k <= 2 * r; k++) sum += w.emplace_back(std::pow(2.0f, -((k - r) * (k - r)) / ((R / 3) * (R / 3))));
            float thr = w[int(0.375f * R + 0.5f)] / sum;
            // Note: exact Gaussians at every radius (a box approximation would be cheaper above radius 32); swap if slow at 4K
            V = sepBlur(adjust(V, sepBlur(g0, R), thr * thr), 2.5f * R);
        }
        V = adjust(sepBlur(adjust(V, g0, 0), 5), g0, 0);
        sc << V;
        return V;
    }

    // ---- Straight <-> premultiplied (AE frames are straight) ----

    RMBFrame premultiplied(const RMBFrame& f)
    {
        RMBFrame out = { Buf(dev, (NSUInteger)f.width * f.height * 16), f.width, f.width, f.height };
        Frame sf = { f.pitch, f.width, f.height }, df = { out.pitch, out.width, out.height };
        [enc setBuffer:f.buf offset:0 atIndex:0];
        [enc setBytes:&sf length:sizeof sf atIndex:1];
        [enc setBuffer:out.buf offset:0 atIndex:2];
        [enc setBytes:&df length:sizeof df atIndex:3];
        Dispatch(enc, ctx["premultiply"], f.width, f.height);
        return out;
    }
    void unpremultiply(const RMBFrame& f)
    {
        Frame ff = { f.pitch, f.width, f.height };
        [enc setBuffer:f.buf offset:0 atIndex:0];
        [enc setBytes:&ff length:sizeof ff atIndex:1];
        Dispatch(enc, ctx["unpremultiply"], f.width, f.height);
    }

    // One flow problem. cov1/cov0: binarised coverage pyramids of cur/other for layer tracking (nullptr: fully
    // covered).
    // guide: guide pyramid (guidePyramid) pinning guided pixels, or nullptr.
    struct Problem {
        const Pyr* I1; const Pyr* I0; int L0; const Pyr* cov1 = nullptr; const Pyr* cov0 = nullptr;
        const Pyr* guide = nullptr;
        bool parallelFine = false;  // New: parallel propagation instead of the serial block match at level 0
    };

    id<MTLTexture> solve(const Pyr& I1, const Pyr& I0, int L0, const Pyr* cov1 = nullptr, const Pyr* cov0 = nullptr)
    {
        return solveMany({ { &I1, &I0, L0, cov1, cov0 } })[0];
    }

    // Independent flows solved in lockstep: each step is dispatched for every problem with no barrier in between,
    // so their (mostly serial) block-match sweeps run side by side on the GPU. Same maths as solving them one by one.
    std::vector<id<MTLTexture>> solveMany(const std::vector<Problem>& P)
    {
        ones();
#ifdef RMB_PROFILE
        if (gProf) stage("solve", true); else
#endif
        { [enc endEncoding]; enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; }
        auto barrier = [&] { [enc memoryBarrierWithScope:MTLBarrierScopeTextures | MTLBarrierScopeBuffers]; };

        struct St { std::vector<id<MTLTexture>> disp; id<MTLBuffer> check; id<MTLTexture> grad, A, b, bw, tmp, cost, c1, c0; int useCov, shared; };
        std::vector<St> st(P.size());
        int Lmax = 0;
        for (size_t k = 0; k < P.size(); k++) {
            const Problem& p = P[k];
            Lmax = std::max(Lmax, p.L0);
            uint32_t init[2] = { 1, 0 };  // { still checking, any vector inside }
            st[k].check = [dev newBufferWithBytes:init length:sizeof init options:MTLResourceStorageModeShared];
            st[k].disp.resize(p.I1->size());
            for (int L = 0; L <= p.L0; L++)
                st[k].disp[L] = Tex(dev, (*p.I1)[L].width, (*p.I1)[L].height, MTLPixelFormatRG32Float);
            [enc setTexture:st[k].disp[p.L0] atIndex:0];
            Dispatch(enc, ctx["legacyClear"], st[k].disp[p.L0].width, st[k].disp[p.L0].height);
        }
        barrier();

        auto each = [&](int L, auto&& fn) {  // run fn(k) for every problem active at level L, then a barrier
            for (size_t k = 0; k < P.size(); k++)
                if (L <= P[k].L0) fn(k);
            barrier();
        };
        bool anyGuide = std::any_of(P.begin(), P.end(), [](const Problem& p) { return p.guide; });
        auto guide = [&](int L) {  // guided pixels take the guides' displacement: disp -> tmp -> disp (idempotent)
            if (!anyGuide) return;
            for (int pass = 0; pass < 2; pass++)
                each(L, [&](size_t k) {
                    if (!P[k].guide) return;
                    St& t = st[k];
                    guideAdjust(pass ? t.tmp : t.disp[L], (*P[k].guide)[L], 0, pass ? t.disp[L] : t.tmp);
                });
        };
        bool anyParallel = std::any_of(P.begin(), P.end(), [](const Problem& p) { return p.parallelFine; });
        auto propagate = [&](int L) {  // New: parallel propagation (newPropagate) at level 0, disp -> tmp -> ... -> disp
            if (!anyParallel) return;
            const int steps[] = { 4, 2, 1, 1 };  // even count: ends back in disp
            for (size_t i = 0; i < 4; i++)
                each(L, [&](size_t k) {
                    if (!P[k].parallelFine) return;
                    St& t = st[k];
                    int step = steps[i];
                    [enc setTexture:(*P[k].I1)[L] atIndex:0];
                    [enc setTexture:(*P[k].I0)[L] atIndex:1];
                    [enc setTexture:i & 1 ? t.tmp : t.disp[L] atIndex:2];
                    [enc setTexture:i & 1 ? t.disp[L] : t.tmp atIndex:3];
                    [enc setTexture:onesTex atIndex:4];
                    [enc setBytes:&step length:sizeof step atIndex:0];
                    Dispatch(enc, ctx["newPropagate"], t.tmp.width, t.tmp.height);
                });
        };
        auto blockMatch = [&](int L) {
            for (int dir = 0; dir < 4; dir++) {  // ping-pong ends back in disp
                each(L, [&](size_t k) {
                    if (L == 0 && P[k].parallelFine) return;
                    St& t = st[k];
                    [enc setTexture:(*P[k].I1)[L] atIndex:0];
                    [enc setTexture:(*P[k].I0)[L] atIndex:1];
                    [enc setTexture:dir & 1 ? t.tmp : t.disp[L] atIndex:2];
                    [enc setTexture:t.cost atIndex:3];
                    [enc setTexture:t.c1 atIndex:4];
                    [enc setTexture:t.c0 atIndex:5];
                    [enc setBytes:&dir length:sizeof dir atIndex:0];
                    [enc setBytes:&t.useCov length:sizeof t.useCov atIndex:1];
                    Dispatch(enc, ctx["legacySweepCosts"], t.cost.width, t.cost.height);
                });
                each(L, [&](size_t k) {
                    if (L == 0 && P[k].parallelFine) return;
                    St& t = st[k];
                    [enc setTexture:(*P[k].I1)[L] atIndex:0];
                    [enc setTexture:(*P[k].I0)[L] atIndex:1];
                    [enc setTexture:dir & 1 ? t.tmp : t.disp[L] atIndex:2];
                    [enc setTexture:dir & 1 ? t.disp[L] : t.tmp atIndex:3];
                    [enc setTexture:t.cost atIndex:4];
                    [enc setTexture:t.c0 atIndex:5];
                    [enc setBytes:&dir length:sizeof dir atIndex:0];
                    [enc setBytes:&t.useCov length:sizeof t.useCov atIndex:1];
                    Dispatch(enc, ctx["legacyBlockMatch"], (dir < 2 ? t.cost.height : t.cost.width) - 2, 1);
                });
            }
        };

        for (int L = Lmax; L >= 0; L--) {
            Scope level;  // this level's temporaries are reusable by the next (finer) level
            for (auto& t : st) for (auto d : t.disp) if (d) level << d;
            each(L, [&](size_t k) {
                St& t = st[k];
                const Problem& p = P[k];
                NSUInteger W = (*p.I1)[L].width, H = (*p.I1)[L].height;
                t.shared = -1;  // same current image, no coverage: reuse an earlier problem's gradient and A
                for (size_t j = 0; j < k && t.shared < 0; j++)
                    if (P[j].I1 == p.I1 && !P[j].cov1 && !p.cov1 && L <= P[j].L0) t.shared = int(j);
                t.c1 = p.cov1 ? (*p.cov1)[L] : onesTex;
                t.c0 = p.cov0 ? (*p.cov0)[L] : onesTex;
                t.useCov = p.cov1 != nullptr;
                t.b = Tex(dev, W, H, MTLPixelFormatRG32Float);
                t.tmp = Tex(dev, W, H, MTLPixelFormatRG32Float);
                t.cost = Tex(dev, W, H, MTLPixelFormatRG32Float);
                if (t.shared >= 0) { t.grad = st[t.shared].grad; return; }
                t.grad = Tex(dev, W, H, MTLPixelFormatRG32Float);
                t.A = Tex(dev, W, H, MTLPixelFormatRGBA32Float);
                [enc setTexture:(*p.I1)[L] atIndex:0];
                [enc setTexture:t.grad atIndex:1];
                [enc setTexture:t.A atIndex:2];
                [enc setTexture:t.c1 atIndex:3];
                [enc setBytes:&t.useCov length:sizeof t.useCov atIndex:1];
                Dispatch(enc, ctx["legacyGradAndA"], W, H);
            });
            guide(L);
            each(L, [&](size_t k) { st[k].A = st[k].shared >= 0 ? st[st[k].shared].A : filtered(st[k].A, kGauss7); });

            for (int it = 0; it < 3; it++) {
                each(L, [&](size_t k) {
                    St& t = st[k];
                    [enc setTexture:(*P[k].I1)[L] atIndex:0];
                    [enc setTexture:(*P[k].I0)[L] atIndex:1];
                    [enc setTexture:t.grad atIndex:2];
                    [enc setTexture:t.disp[L] atIndex:3];
                    [enc setTexture:t.b atIndex:4];
                    [enc setTexture:t.c0 atIndex:5];
                    [enc setBytes:&t.useCov length:sizeof t.useCov atIndex:1];
                    Dispatch(enc, ctx["legacyCalcB"], t.b.width, t.b.height);
                });
                each(L, [&](size_t k) { st[k].bw = filtered(st[k].b, kGauss7); });
                each(L, [&](size_t k) {
                    St& t = st[k];
                    [enc setTexture:t.disp[L] atIndex:0];
                    [enc setTexture:t.A atIndex:1];
                    [enc setTexture:t.bw atIndex:2];
                    [enc setTexture:t.tmp atIndex:3];
                    [enc setBytes:&kNewDispEpsilon length:sizeof(float) atIndex:0];
                    [enc setBytes:&kMaxStep length:sizeof(float) atIndex:1];
                    Dispatch(enc, ctx["legacyNewDisp"], t.b.width, t.b.height);
                });
                each(L, [&](size_t k) { filter(st[k].tmp, st[k].disp[L], kGauss5); });
                if (it == 1 && L == 0) { blockMatch(L); propagate(L); guide(L); }
            }
            if (L != 0) blockMatch(L);
            guide(L);

            for (const char* kname : { "legacyAnyInside", "legacyZeroIfAllOutside", "legacyUpdateCheck" })
                each(L, [&](size_t k) {
                    if (P[k].guide) return;  // guided flows are never reset
                    St& t = st[k];
                    [enc setTexture:t.disp[L] atIndex:0];
                    [enc setBuffer:t.check offset:0 atIndex:0];
                    bool one = std::string(kname) == "legacyUpdateCheck";
                    Dispatch(enc, ctx[kname], one ? 1 : t.disp[L].width, one ? 1 : t.disp[L].height);
                });

            if (L > 0)  // corner-aligned, vectors scaled by the size ratio
                each(L, [&](size_t k) {
                    id<MTLTexture> from = st[k].disp[L], to = st[k].disp[L - 1];
                    resample(from, to, simd_make_float4((to.width - 1.0f) / std::max<float>(1, from.width - 1.0f),
                                                        (to.height - 1.0f) / std::max<float>(1, from.height - 1.0f), 1, 1));
                });
        }
#ifdef RMB_PROFILE
        if (gProf) stage("after solve"); else
#endif
        { [enc endEncoding]; enc = [cb computeCommandEncoder]; }
        std::vector<id<MTLTexture>> out;
        for (auto& t : st) out.push_back(t.disp[0]);
        return out;
    }

    // ---- Layer helpers (foreground matte) ----

    id<MTLTexture> matteChannel(const RMBFrame& m, int chan)
    {
        id<MTLTexture> out = Tex(dev, m.width, m.height, MTLPixelFormatR32Float);
        Frame f = { m.pitch, m.width, m.height };
        [enc setBuffer:m.buf offset:0 atIndex:0];
        [enc setBytes:&f length:sizeof f atIndex:1];
        [enc setBytes:&chan length:sizeof chan atIndex:2];
        [enc setTexture:out atIndex:0];
        Dispatch(enc, ctx["layerMatteChannel"], m.width, m.height);
        return out;
    }
    id<MTLTexture> sepBlur(id<MTLTexture> in, float R)  // taps over +/-floor(R), w = 2^(-i^2/(R/3)^2)
    {
        int r = int(std::floor(R));
        if (r < 1) return in;
        std::vector<float> w;
        for (int i = -r; i <= r; i++) w.push_back(std::pow(2.0f, -(i * i) / ((R / 3) * (R / 3))));
        id<MTLTexture> t = in;
        for (int v = 0; v < 2; v++) {
            id<MTLTexture> out = Tex(dev, in.width, in.height, in.pixelFormat);
            [enc setTexture:t atIndex:0];
            [enc setTexture:out atIndex:1];
            if (w.size() * sizeof(float) <= 4096) [enc setBytes:w.data() length:w.size() * sizeof(float) atIndex:0];
            else [enc setBuffer:[dev newBufferWithBytes:w.data() length:w.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared] offset:0 atIndex:0];
            [enc setBytes:&r length:sizeof r atIndex:1];
            [enc setBytes:&v length:sizeof v atIndex:2];
            Dispatch(enc, ctx["layerSepFilter"], in.width, in.height);
            t = out;
        }
        return t;
    }
    id<MTLTexture> chokeInt(id<MTLTexture> x, float m)
    {
        float c = std::ceil(m + 1), f = c - (m + 1);
        if (f < 0.01f) { c = std::ceil(m + 1.01f); f = c - (m + 1.01f); }
        id<MTLTexture> blurred = sepBlur(x, c + 1 - f);
        std::vector<float> w;
        float sum = 0, s2 = 0, s3 = 0;
        for (int k = 0; k <= 2 * int(c); k++) { w.push_back(std::pow(2.0f, -((k - c) * (k - c)) / ((c / 3) * (c / 3)))); sum += w.back(); }
        for (int k = 0; k < int(w.size()); k++) { if (k >= 2) s2 += w[k] / sum; if (k >= 3) s3 += w[k] / sum; }
        float mn = s3 + (s2 - s3) * (1 - f);
        id<MTLTexture> out = Tex(dev, x.width, x.height, x.pixelFormat);
        [enc setTexture:blurred atIndex:0];
        [enc setTexture:out atIndex:1];
        [enc setBytes:&mn length:sizeof mn atIndex:0];
        Dispatch(enc, ctx["layerChokeMap"], x.width, x.height);
        return out;
    }
    id<MTLTexture> blend(id<MTLTexture> a, id<MTLTexture> b, float f, bool clampOut)
    {
        id<MTLTexture> out = Tex(dev, a.width, a.height, a.pixelFormat);
        int c = clampOut;
        [enc setTexture:a atIndex:0];
        [enc setTexture:b atIndex:1];
        [enc setTexture:out atIndex:2];
        [enc setBytes:&f length:sizeof f atIndex:0];
        [enc setBytes:&c length:sizeof c atIndex:1];
        Dispatch(enc, ctx["layerBlend"], a.width, a.height);
        return out;
    }
    // Matte choke by `a` pixels; negative grows (invert, choke, invert). Fractional sizes blend floor and ceil.
    id<MTLTexture> choke(id<MTLTexture> m, float a)
    {
        if (std::fabs(a) < 1e-4f) return m;
        Scope sc;
        bool inv = a < 0;
        float sz = std::fabs(a), fr = sz - std::floor(sz);
        id<MTLTexture> x = inv ? unary("layerChokeInvert", m) : m;
        id<MTLTexture> r = fr < 1e-4f ? chokeInt(x, sz)
                                      : blend(chokeInt(x, std::floor(sz)), chokeInt(x, std::ceil(sz)), fr, false);
        if (inv) r = unary("layerChokeInvert", r);
        r = blend(r, r, 0, true);  // clamp to 0..1
        sc << r;
        return r;
    }
    id<MTLTexture> behind(id<MTLTexture> raw, id<MTLTexture> front)
    {
        id<MTLTexture> out = Tex(dev, front.width, front.height, MTLPixelFormatR32Float);
        [enc setTexture:raw atIndex:0];
        [enc setTexture:front atIndex:1];
        [enc setTexture:out atIndex:2];
        Dispatch(enc, ctx["layerBehindMatte"], out.width, out.height);
        return out;
    }
    id<MTLTexture> layerColor(const RMBFrame& src, id<MTLTexture> matte)
    {
        id<MTLTexture> out = Tex(dev, src.width, src.height, MTLPixelFormatRGBA32Float);
        Frame f = { src.pitch, src.width, src.height };
        [enc setBuffer:src.buf offset:0 atIndex:0];
        [enc setBytes:&f length:sizeof f atIndex:1];
        [enc setTexture:matte atIndex:0];
        [enc setTexture:out atIndex:1];
        Dispatch(enc, ctx["layerColor"], src.width, src.height);
        return out;
    }
    id<MTLTexture> warp(id<MTLTexture> cn, id<MTLTexture> d)
    {
        id<MTLTexture> out = Tex(dev, cn.width, cn.height, cn.pixelFormat);
        simd_float2 ratio = { float(d.width) / cn.width, float(d.height) / cn.height };
        [enc setTexture:cn atIndex:0];
        [enc setTexture:d atIndex:1];
        [enc setTexture:out atIndex:2];
        [enc setBytes:&ratio length:sizeof ratio atIndex:0];
        Dispatch(enc, ctx["layerWarp"], out.width, out.height);
        return out;
    }
    id<MTLTexture> over(id<MTLTexture> dst, id<MTLTexture> src)
    {
        id<MTLTexture> out = Tex(dev, dst.width, dst.height, dst.pixelFormat);
        [enc setTexture:dst atIndex:0];
        [enc setTexture:src atIndex:1];
        [enc setTexture:out atIndex:2];
        Dispatch(enc, ctx["layerOver"], out.width, out.height);
        return out;
    }
    // Push-pull fill of everything that is not fully opaque; returns the unpremultiplied, filled layer.
    id<MTLTexture> fill(id<MTLTexture> premult)
    {
        Scope sc;
        id<MTLTexture> B0 = Tex(dev, premult.width, premult.height, MTLPixelFormatRGBA32Float);
        id<MTLTexture> C0 = Tex(dev, premult.width, premult.height, MTLPixelFormatR32Float);
        [enc setTexture:premult atIndex:0];
        [enc setTexture:B0 atIndex:1];
        [enc setTexture:C0 atIndex:2];
        Dispatch(enc, ctx["layerFillSeed"], premult.width, premult.height);
        id<MTLTexture> A = pushPull(B0, C0);
        sc << A;
        return A;
    }
    // Push-pull: B0 = values * valid (rgba32Float), C0 = valid (r32Float). Returns B/C where valid, smoothly
    // extrapolated from the valid samples elsewhere.
    id<MTLTexture> pushPull(id<MTLTexture> B0, id<MTLTexture> C0)
    {
        Scope sc;
        Pyr B = { B0 }, C = { C0 };
        auto push = [&](id<MTLTexture> t) {
            id<MTLTexture> f = unary("layerBinomialV", unary("layerBinomialFlipH", t));
            id<MTLTexture> out = Tex(dev, std::max<NSUInteger>(2, t.width / 2), std::max<NSUInteger>(2, t.height / 2), t.pixelFormat);
            resample(f, out, simd_make_float4(1, 1, 1, 1));
            return out;
        };
        while (!(B.back().width <= 2 && B.back().height <= 2)) { B.push_back(push(B.back())); C.push_back(push(C.back())); }
        id<MTLTexture> A = Tex(dev, B.back().width, B.back().height, MTLPixelFormatRGBA32Float);
        setValue(A, 0);
        auto adjust = [&](id<MTLTexture> a, int l) {
            id<MTLTexture> out = Tex(dev, a.width, a.height, a.pixelFormat);
            [enc setTexture:a atIndex:0];
            [enc setTexture:B[l] atIndex:1];
            [enc setTexture:C[l] atIndex:2];
            [enc setTexture:out atIndex:3];
            Dispatch(enc, ctx["layerFillAdjust"], a.width, a.height);
            return out;
        };
        for (int l = int(B.size()) - 1; l >= 0; l--) {
            A = filtered(adjust(A, l), kGauss5);
            if (l > 0) {
                id<MTLTexture> up = Tex(dev, B[l - 1].width, B[l - 1].height, MTLPixelFormatRGBA32Float);
                resample(A, up, simd_make_float4(1, 1, 1, 1));
                A = up;
            } else {
                A = adjust(A, 0);
            }
        }
        sc << A;
        return A;
    }
    id<MTLTexture> finish(id<MTLTexture> filled, id<MTLTexture> alpha, const RMBFrame& src)
    {
        id<MTLTexture> out = Tex(dev, filled.width, filled.height, MTLPixelFormatRGBA32Float), one = ones();
        Frame f = { src.pitch, src.width, src.height };
        int fromSrc = alpha == nil;
        [enc setTexture:filled atIndex:0];
        [enc setTexture:alpha ?: one atIndex:1];
        [enc setBuffer:src.buf offset:0 atIndex:0];
        [enc setBytes:&f length:sizeof f atIndex:1];
        [enc setBytes:&fromSrc length:sizeof fromSrc atIndex:2];
        [enc setTexture:out atIndex:2];
        Dispatch(enc, ctx["layerFinish"], out.width, out.height);
        return out;
    }
    RMBFrame toBuffer(id<MTLTexture> t)
    {
        RMBFrame fr = { Buf(dev, t.width * t.height * 16),
                        (uint32_t)t.width, (uint32_t)t.width, (uint32_t)t.height };
        Frame f = { fr.pitch, fr.width, fr.height };
        [enc setTexture:t atIndex:0];
        [enc setBuffer:fr.buf offset:0 atIndex:0];
        [enc setBytes:&f length:sizeof f atIndex:1];
        Dispatch(enc, ctx["layerToBuffer"], t.width, t.height);
        return fr;
    }
};

void EncodeLegacyBlur(Ctx& ctx, id<MTLComputeCommandEncoder> enc, const RMBFrame& src, const RMBFrame& dst,
                      id<MTLTexture> flow, float blurAmount, bool hq)
{
    float amount = blurAmount < 0 ? 0.5f : blurAmount;  // negative: blur at 0.5 then extrapolate
    LegacyBlur p;
    p.ratio[0] = float(flow.width) / src.width;
    p.ratio[1] = float(flow.height) / src.height;
    p.scale[0] = 0.5f / p.ratio[0] * amount;
    p.scale[1] = 0.5f / p.ratio[1] * amount;
    p.mix = blurAmount < 0 ? blurAmount : 1.0f;
    p.hq = hq;
    Frame sf = { src.pitch, src.width, src.height }, df = { dst.pitch, dst.width, dst.height };
    [enc setBuffer:src.buf offset:0 atIndex:0];
    [enc setBuffer:dst.buf offset:0 atIndex:1];
    [enc setBytes:&sf length:sizeof sf atIndex:2];
    [enc setBytes:&df length:sizeof df atIndex:3];
    [enc setBytes:&p length:sizeof p atIndex:4];
    [enc setTexture:flow atIndex:0];
    Dispatch(enc, ctx["legacyMotionBlur"], dst.width, dst.height);
}

// ---------------------------------------------------------------------------------------------------------------
// New mode: the Legacy tracker run in both directions (sharing the current frame's pyramid), then a
// depth-bucketed reconstruction blur. Encoded entirely on the GPU.

struct NBlur { uint32_t w, h; float scale, mix; int32_t mode, tile, samples; float softZ, ownDensity; int32_t matteDepth; };
const int kTile = 32;

struct NewEncoder {
    LegacyEncoder& le;

    // With a FG matte, a pixel's velocity comes from its own layer: each layer's field is rebuilt by push-pull from
    // pixels well inside that layer (24 px erosion at 1080p), so neither borrows the other's motion at the edge.
    struct Clean { id<MTLTexture> picked, fg, bg; };
    Clean cleanVelocity(id<MTLTexture> dPrev, id<MTLTexture> dNext, int mode, id<MTLTexture> matte, int halvings)
    {
        id<MTLTexture> ref = mode == 1 ? dNext : dPrev;
        NSUInteger w = ref.width, h = ref.height;
        Scope sc;
        id<MTLTexture> v = Tex(le.dev, w, h, MTLPixelFormatRG32Float);
        [le.enc setTexture:dPrev atIndex:0];
        [le.enc setTexture:dNext atIndex:1];
        [le.enc setTexture:v atIndex:2];
        [le.enc setBytes:&mode length:sizeof mode atIndex:0];
        Dispatch(le.enc, le.ctx["newVelocityLow"], w, h);
        id<MTLTexture> m = matte;
        for (int i = 0; i < halvings; i++) m = le.halve(m);
        id<MTLTexture> mBlur = le.sepBlur(m, std::max(2.0f, 24.0f * w / 1920.0f));  // 24 px at 1080p, in tracking px
        id<MTLTexture> field[2];
        for (int fg = 0; fg < 2; fg++) {
            id<MTLTexture> B = Tex(le.dev, w, h, MTLPixelFormatRGBA32Float), C = Tex(le.dev, w, h, MTLPixelFormatR32Float);
            [le.enc setTexture:v atIndex:0];
            [le.enc setTexture:mBlur atIndex:1];
            [le.enc setTexture:B atIndex:2];
            [le.enc setTexture:C atIndex:3];
            [le.enc setBytes:&fg length:sizeof fg atIndex:0];
            Dispatch(le.enc, le.ctx["newVelocitySeed"], w, h);
            field[fg] = le.pushPull(B, C);
        }
        id<MTLTexture> out = Tex(le.dev, w, h, MTLPixelFormatRG32Float);
        [le.enc setTexture:field[1] atIndex:0];
        [le.enc setTexture:field[0] atIndex:1];
        [le.enc setTexture:m atIndex:2];
        [le.enc setTexture:out atIndex:3];
        Dispatch(le.enc, le.ctx["newVelocityPick"], w, h);
        sc << out << field[1] << field[0];
        return { out, field[1], field[0] };
    }

    // usePrev/useNext: which neighbours to track against. depth: FG matte (nearer = 1) or nil for the layer's alpha.
    void render(const RMBInputs& in, const RMBFrame& dst, const RMBSettings& s, bool usePrev, bool useNext,
                id<MTLTexture> depthTex)
    {
        const RMBFrame& src = in.src[1];
        const RMBFrame& cur = in.track[1];
        // dPrev: cur(x) ~ prev(x - dPrev), i.e. the motion that arrived; dNext: cur(x) ~ next(x - dNext).
        id<MTLTexture> dPrev = nil, dNext = nil;
        LegacyEncoder::Plan pl = LegacyEncoder::plan(cur, s.hq, s.sensitivity);
        RMBGuides g = s.guides[RMBLayerBG];
        const RMBGuides& gf = s.guides[RMBLayerFG];
        g.prev.insert(g.prev.end(), gf.prev.begin(), gf.prev.end());
        g.next.insert(g.next.end(), gf.next.begin(), gf.next.end());
        id<MTLTexture> gPrev = usePrev ? le.guideImage(g.prev, cur, pl) : nil, gNext = useNext ? le.guideImage(g.next, cur, pl) : nil;
        if (!g.imageTrack) {
            dPrev = le.morph(gPrev, pl);
            dNext = le.morph(gNext, pl);
        } else if (s.amount != 0 && s.sensitivity > 0 && (usePrev || useNext)) {
            le.stage("new pyramids");
            auto build = [&](const RMBFrame& f) { return le.pyramidNew(f, pl.halvings, pl.levels, s.trackRGBA); };
            auto I1 = build(cur);
            Pyr Ip = usePrev ? build(in.track[0]) : Pyr();
            Pyr In = useNext ? build(in.track[2]) : Pyr();
            Pyr Gp = gPrev ? le.guidePyramid(gPrev, I1, pl.L0) : Pyr(), Gn = gNext ? le.guidePyramid(gNext, I1, pl.L0) : Pyr();
            std::vector<LegacyEncoder::Problem> P;
            if (usePrev) P.push_back({ &I1, &Ip, pl.L0, nullptr, nullptr, gPrev ? &Gp : nullptr, true });
            if (useNext) P.push_back({ &I1, &In, pl.L0, nullptr, nullptr, gNext ? &Gn : nullptr, true });
            auto d = le.solveMany(P);  // both directions in lockstep
            if (usePrev) dPrev = d[0];
            if (useNext) dNext = d.back();
        }
        id<MTLDevice> dev = le.dev;
        Ctx& ctx = le.ctx;
        le.stage("new velocity + tiles + reconstruct");  // (before taking le.enc: a profiling stage swaps it)
        id<MTLComputeCommandEncoder> enc = le.enc;
        id<MTLTexture> zero = nil;
        if (!dPrev || !dNext) {
            zero = Tex(dev, 1, 1, MTLPixelFormatRG32Float);
            [enc setTexture:zero atIndex:0];
            Dispatch(enc, ctx["legacyClear"], 1, 1);
        }
        int mode = dPrev && dNext ? 0 : dNext ? 1 : 2;
        bool clean = true;
#ifdef RMB_PROFILE
        clean = !getenv("RMB_NO_CLEAN");
#endif
        Clean cv = {};
        if (clean && depthTex && (dPrev || dNext)) {
            cv = cleanVelocity(dPrev ?: zero, dNext ?: zero, mode, depthTex, pl.halvings);
            dPrev = cv.picked;
            mode = 2;  // the cleaned per-frame velocity is passed as "previous only"
        }

        NBlur b = { src.width, src.height, 0.5f * (s.amount < 0 ? 0.5f : s.amount), s.amount < 0 ? s.amount : 1.0f,
                    mode, kTile, 16, 0.5f, 1.0f, depthTex != nil };  // depth: matte, else alpha
        Frame sf = { src.pitch, src.width, src.height }, df = { dst.pitch, dst.width, dst.height };
        uint32_t tw = (src.width + kTile - 1) / kTile, th = (src.height + kTile - 1) / kTile;
        id<MTLBuffer> tmax = Buf(dev, tw * th * 16), nmax = Buf(dev, tw * th * 16);
        id<MTLTexture> one = le.ones();

        // Motion is read straight from the tracker's fields and depth from the matte or alpha (no full-res copies).
        [enc setTexture:dPrev ?: zero atIndex:0];
        [enc setTexture:dNext ?: zero atIndex:1];
        [enc setTexture:depthTex ?: one atIndex:2];
        [enc setBuffer:tmax offset:0 atIndex:1];
        [enc setBytes:&b length:sizeof b atIndex:2];
        [enc setBuffer:src.buf offset:0 atIndex:4];
        [enc setBytes:&sf length:sizeof sf atIndex:5];
        [enc setComputePipelineState:ctx["newTileMax"]];  // one 32x8 threadgroup per tile
        [enc dispatchThreadgroups:MTLSizeMake(tw, th, 1) threadsPerThreadgroup:MTLSizeMake(kTile, 8, 1)];
        [enc setBuffer:tmax offset:0 atIndex:0];
        [enc setBuffer:nmax offset:0 atIndex:1];
        Dispatch(enc, ctx["newNeighborMax"], tw, th);
        [enc setBuffer:src.buf offset:0 atIndex:0];
        [enc setBuffer:dst.buf offset:0 atIndex:1];
        [enc setBytes:&sf length:sizeof sf atIndex:2];
        [enc setBytes:&df length:sizeof df atIndex:3];
        [enc setBytes:&b length:sizeof b atIndex:4];
        [enc setBuffer:nmax offset:0 atIndex:7];
        Dispatch(enc, ctx["newReconstruct"], dst.width, dst.height);
        if (cv.fg) {
            // Background pixels: two layers, each along its own motion: FG (colour * matte) over the background
            // plate (the FG's hole filled by push-pull). FG pixels keep the depth-aware reconstruction above.
            id<MTLTexture> plate = le.finish(le.fill(le.layerColor(src, le.unary("layerChokeInvert", depthTex))), nil, src);
            [enc setBuffer:src.buf offset:0 atIndex:0];
            [enc setBuffer:dst.buf offset:0 atIndex:1];
            [enc setBytes:&sf length:sizeof sf atIndex:2];
            [enc setBytes:&df length:sizeof df atIndex:3];
            [enc setBytes:&b length:sizeof b atIndex:4];
            [enc setTexture:cv.fg atIndex:0];
            [enc setTexture:cv.bg atIndex:1];
            [enc setTexture:depthTex atIndex:2];
            [enc setTexture:plate atIndex:3];
            Dispatch(enc, ctx["newLayered"], dst.width, dst.height);
        }
    }
};

bool Usable(const RMBFrame& f, const RMBFrame& ref) { return f.buf && f.width == ref.width && f.height == ref.height; }

// Mean |luma(a) - luma(b)| and |luma(a) - luma(c)| in 0..1, in one pass and one read-back. A duplicate frame (0)
// counts as maximally different.
std::pair<float, float> FrameDiffs(id<MTLDevice> dev, id<MTLCommandQueue> q, Ctx& ctx, const RMBFrame& a,
                                   const RMBFrame& b, const RMBFrame& c)
{
    id<MTLBuffer> sum = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
    memset(sum.contents, 0, 8);
    Frame fa = { a.pitch, a.width, a.height }, fb = { b.pitch, b.width, b.height }, fc = { c.pitch, c.width, c.height };
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setBuffer:a.buf offset:0 atIndex:0];
    [enc setBuffer:b.buf offset:0 atIndex:1];
    [enc setBuffer:c.buf offset:0 atIndex:2];
    [enc setBytes:&fa length:sizeof fa atIndex:3];
    [enc setBytes:&fb length:sizeof fb atIndex:4];
    [enc setBytes:&fc length:sizeof fc atIndex:5];
    [enc setBuffer:sum offset:0 atIndex:6];
    Dispatch(enc, ctx["frameDiff"], a.width, a.height);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* r = (const float*)sum.contents;
    float n = float(a.width) * a.height, db = r[0] / n, dc = r[1] / n;
    return { db == 0 ? 1.0f : db, dc == 0 ? 1.0f : dc };
}

const float kCutThreshold = 0.06f;  // mean luma difference above which a neighbour may be across a cut

// Legacy layers. Each is tracked on its own matted images (both layers' flows are solved together), then its
// colour is rebuilt (current over the warped neighbour, holes filled) and blurred with its own amount.
struct LayerJob {
    id<MTLTexture> layT, layN, covT, covN, alpha;  // covT/covN nil: fully covered (front layer); alpha nil: source alpha
    float amount, sensitivity;
    const RMBGuides* guides;
    Pyr I1, I0, c1, c0, G;
    LegacyEncoder::Plan plan;
    id<MTLTexture> d;
};

void LegacyLayerFlows(LegacyEncoder& le, const RMBInputs& in, int n, std::vector<LayerJob*> jobs, const RMBSettings& s)
{
    le.stage("layer pyramids + coverage");
    std::vector<LegacyEncoder::Problem> P;
    std::vector<LayerJob*> tracked;
    for (LayerJob* j : jobs) {
        j->plan = LegacyEncoder::plan(in.track[1], s.hq, j->sensitivity);
        if (j->amount == 0 || j->sensitivity <= 0) {
            j->d = le.zeroFlow();
            continue;
        }
        id<MTLTexture> g0 = le.guideImage(n == 0 ? j->guides->prev : j->guides->next, in.track[1], j->plan);
        if (!j->guides->imageTrack) {
            j->d = le.morph(g0, j->plan) ?: le.zeroFlow();
            continue;
        }
        j->I1 = le.pyramid(in.track[1], j->plan.halvings, j->plan.levels, s.trackRGBA, j->layT);
        j->I0 = le.pyramid(in.track[n], j->plan.halvings, j->plan.levels, s.trackRGBA, j->layN);
        if (g0) j->G = le.guidePyramid(g0, j->I1, j->plan.L0);
        if (j->covT) {
            j->c1 = le.coverage(j->covT, j->plan.halvings, j->plan.levels);
            j->c0 = le.coverage(j->covN, j->plan.halvings, j->plan.levels);
        }
        P.push_back({ &j->I1, &j->I0, j->plan.L0, j->covT ? &j->c1 : nullptr, j->covT ? &j->c0 : nullptr,
                      j->G.empty() ? nullptr : &j->G });
        tracked.push_back(j);
    }
    if (P.empty()) return;
    auto d = le.solveMany(P);
    for (size_t k = 0; k < tracked.size(); k++) tracked[k]->d = d[k];
    for (LayerJob* j : tracked) j->I1 = j->I0 = j->c1 = j->c0 = j->G = Pyr();
}

RMBFrame LegacyLayerBlur(LegacyEncoder& le, const RMBInputs& in, int n, const LayerJob& j, const RMBSettings& s)
{
    Scope sc;  // everything but the blurred layer is reusable afterwards
    const RMBFrame& src = in.src[1];
    const RMBFrame& srcN = Usable(in.src[n], src) ? in.src[n] : src;
    le.stage("layer colour + warp + over");
    id<MTLTexture> ct = le.layerColor(src, j.layT);
    id<MTLTexture> w = le.over(le.warp(le.layerColor(srcN, j.layN), j.d), ct);
    le.stage("layer fill");
    id<MTLTexture> filled = le.fill(w);
    le.stage("layer finish + blur");
    RMBFrame layer = le.toBuffer(le.finish(filled, j.alpha, src));
    RMBFrame blurred = { Buf(le.dev, layer.buf.length),
                         layer.pitch, layer.width, layer.height };
    EncodeLegacyBlur(le.ctx, le.enc, layer, blurred, j.d, j.amount, s.hq);
    sc << blurred.buf;
    return blurred;
}

}  // namespace

void RMBLog(const char* fmt, ...)
{
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    NSString* path = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Logs/RealMotionBlur.log"];
    FILE* f = fopen(path.fileSystemRepresentation, "a");
    if (!f) return;
    char when[32];
    time_t now = time(nullptr);
    strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", localtime(&now));
    fprintf(f, "%s ", when);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

bool RMBRender(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBInputs& inRaw, const RMBFrame& dst,
               RMBSettings s, bool wait)
{
    @autoreleasepool {
        Ctx* ctx = ContextFor(device);
        RMBInputs in = inRaw;
        const RMBFrame& src = in.src[1];
        if (!ctx || !src.buf || !dst.buf) {
            RMBLog("render: %s", !ctx ? "no Metal kernels" : !src.buf ? "no input frame" : "no output frame");
            return false;
        }
        if (!in.track[1].buf) for (int i = 0; i < 3; i++) in.track[i] = in.src[i];  // no Motion Src: track the layer
        bool hasPrev = Usable(in.track[0], in.track[1]), hasNext = Usable(in.track[2], in.track[1]);
        bool matte = Usable(in.matte[1], src) && s.display != RMBDisplayUnblurred;

        // Which neighbours to track against (Track Frame). Auto skips a neighbour that looks like it is across a cut.
        bool usePrev = hasPrev, useNext = hasNext;
        if (s.trackFrame == RMBTrackNext && hasNext) usePrev = false;
        if (s.trackFrame == RMBTrackPrevious && hasPrev) useNext = false;
        if (s.trackFrame == RMBTrackAuto && !s.guides[RMBLayerBG].imageTrack && hasPrev) useNext = false;  // without image tracking only the previous frame is used
        else if (s.trackFrame == RMBTrackAuto && hasPrev && hasNext && s.amount != 0) {
            auto [dP, dN] = FrameDiffs(device, queue, *ctx, in.track[1], in.track[0], in.track[2]);
            if (s.legacy) {  // one neighbour: previous unless it differs more than the next
                usePrev = dP <= kCutThreshold || dP < dN;
                useNext = !usePrev;
            } else {         // both, minus whichever side looks like a cut
                usePrev = !(dP > kCutThreshold && dP > dN);
                useNext = !(dN > kCutThreshold && dN > dP);
            }
        }
        if (s.legacy && usePrev && useNext) useNext = false;  // Legacy tracks a single neighbour (previous first)
        int n = usePrev ? 0 : 2;

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        auto* lease = new Lease();
        tLease = lease;
        LegacyEncoder le = { device, *ctx, [cb computeCommandEncoder], cb };
        le.delinearize = s.trackDelinearize;
        // Nothing to blur: the input as given, bit for bit.
        bool still = s.display == RMBDisplayBlurred && s.amount == 0 && (!matte || !s.legacy || s.fgAmount == 0);
        bool copy = s.display == RMBDisplayUnblurred || !(usePrev || useNext) || still;
        le.stage("premultiply");
        if (!copy) {  // blur premultiplied (tracking stays on the frames as given)
            in.src[1] = le.premultiplied(in.src[1]);
            if (s.legacy && matte && Usable(in.src[n], src)) in.src[n] = le.premultiplied(in.src[n]);
        }
        if (copy) {
            EncodeLegacyBlur(*ctx, le.enc, src, dst, le.zeroFlow(), 0, s.hq);  // zero motion: exact copy
        } else if (!s.legacy) {
            NewEncoder ne = { le };
            ne.render(in, dst, s, usePrev, useNext, matte ? le.matteChannel(in.matte[1], s.matteChannel) : nil);
        } else if (!matte) {
            const RMBGuides& g = s.guides[RMBLayerBG];
            le.stage("legacy pyramids");
            id<MTLTexture> flow = le.flow(in.track[1], in.track[n], s, n == 0 ? &g.prev : &g.next, g.imageTrack);
            le.stage("legacy blur");
            EncodeLegacyBlur(*ctx, le.enc, src, dst, flow, s.amount, s.hq);
        } else {
            // Layered: FG1 = the matte; the background is everything outside the (grown) FG matte.
            le.stage("mattes + choke");
            const RMBFrame& mN = Usable(in.matte[n], src) ? in.matte[n] : in.matte[1];
            id<MTLTexture> mT = le.matteChannel(in.matte[1], s.matteChannel), mNtex = le.matteChannel(mN, s.matteChannel);
            id<MTLTexture> full = Tex(device, src.width, src.height, MTLPixelFormatR32Float);
            le.setValue(full, 1);
            id<MTLTexture> bgT = le.behind(full, le.choke(mT, -s.matteShrink));
            id<MTLTexture> bgN = le.behind(full, le.choke(mNtex, -s.matteShrink));
            LayerJob bgJob = { bgT, bgN, bgT, bgN, nil, s.amount, s.sensitivity, &s.guides[RMBLayerBG] };
            LayerJob fgJob = { mT, mNtex, nil, nil, mT, s.fgAmount, s.fgSensitivity, &s.guides[RMBLayerFG] };
            std::vector<LayerJob*> jobs;
            if (s.display != RMBDisplayFG) jobs.push_back(&bgJob);
            if (s.display != RMBDisplayBG) jobs.push_back(&fgJob);
            {
                Scope flows;  // pyramids and coverage are done once the flows are solved
                LegacyLayerFlows(le, in, n, jobs, s);
                for (LayerJob* j : jobs) flows << j->d;
            }
            RMBFrame bg = {}, fg = {};
            if (s.display != RMBDisplayFG) bg = LegacyLayerBlur(le, in, n, bgJob, s);
            if (s.display != RMBDisplayBG) fg = LegacyLayerBlur(le, in, n, fgJob, s);
            le.stage("composite");
            Frame lf = { src.width, src.width, src.height }, df = { dst.pitch, dst.width, dst.height };
            int mode = s.display;
            [le.enc setBuffer:(bg.buf ?: fg.buf) offset:0 atIndex:0];
            [le.enc setBuffer:(fg.buf ?: bg.buf) offset:0 atIndex:1];
            [le.enc setBuffer:dst.buf offset:0 atIndex:2];
            [le.enc setBytes:&lf length:sizeof lf atIndex:3];
            [le.enc setBytes:&df length:sizeof df atIndex:4];
            [le.enc setBytes:&mode length:sizeof mode atIndex:5];
            Dispatch(le.enc, (*ctx)["layerCompositeOut"], dst.width, dst.height);
        }
        le.stage("unpremultiply");
        if (!copy) le.unpremultiply(dst);
        id<MTLBuffer> bad = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        *(uint32_t*)bad.contents = 0;
        {
            Frame df = { dst.pitch, dst.width, dst.height };
            [le.enc setBuffer:dst.buf offset:0 atIndex:0];
            [le.enc setBytes:&df length:sizeof df atIndex:1];
            [le.enc setBuffer:bad offset:0 atIndex:2];
            Dispatch(le.enc, (*ctx)["finiteGuard"], dst.width, dst.height);
        }
        [le.enc endEncoding];
        tLease = nullptr;
#ifdef RMB_PROFILE
        if (gProf) {
            size_t total = 0;
            for (id<MTLResource> r : lease->all) total += Bytes(r);
            printf("    scratch leased: %.0f MB in %zu resources\n", total / 1048576.0, lease->all.size());
        }
#endif
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
            if (c.error) RMBLog("GPU command buffer failed: %s", c.error.localizedDescription.UTF8String);
            if (uint32_t n = *(const uint32_t*)bad.contents) RMBLog("render: %u non-finite output pixels zeroed", n);
            ReturnToPool(lease);
        }];
        [cb commit];
        if (wait) [cb waitUntilCompleted];
        return cb.error == nil;
    }
}

bool RMBRenderVectors(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBFrame& src, const RMBFrame& vectors,
                      const RMBFrame& dst, const RMBVectorSettings& s, bool wait)
{
    @autoreleasepool {
        Ctx* ctx = ContextFor(device);
        if (!ctx || !src.buf || !dst.buf) {
            RMBLog("render: %s", !ctx ? "no Metal kernels" : !src.buf ? "no input frame" : "no output frame");
            return false;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        auto* lease = new Lease();
        tLease = lease;
        LegacyEncoder le = { device, *ctx, [cb computeCommandEncoder], cb };
        if (s.amount == 0 || !Usable(vectors, src)) {
            EncodeLegacyBlur(*ctx, le.enc, src, dst, le.zeroFlow(), 0, s.hq);  // no vectors: pass through
        } else {
            id<MTLTexture> B = Tex(device, src.width, src.height, MTLPixelFormatRGBA32Float);
            id<MTLTexture> C = Tex(device, src.width, src.height, MTLPixelFormatR32Float);
            Frame f = { vectors.pitch, vectors.width, vectors.height };
            [le.enc setBuffer:vectors.buf offset:0 atIndex:0];
            [le.enc setBytes:&f length:sizeof f atIndex:1];
            [le.enc setBytes:&s.maxDisplace length:sizeof s.maxDisplace atIndex:2];
            int linear = s.linear;
            [le.enc setBytes:&linear length:sizeof linear atIndex:3];
            [le.enc setTexture:B atIndex:0];
            [le.enc setTexture:C atIndex:1];
            Dispatch(le.enc, (*ctx)["vectorSeed"], src.width, src.height);
            id<MTLTexture> flow = Tex(device, src.width, src.height, MTLPixelFormatRG32Float);
            le.resample(le.pushPull(B, C), flow, simd_make_float4(s.scaleX, s.scaleY, 1, 1));  // holes filled, scaled
            EncodeLegacyBlur(*ctx, le.enc, le.premultiplied(src), dst, flow, s.amount, s.hq);
            le.unpremultiply(dst);
        }
        [le.enc endEncoding];
        tLease = nullptr;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
            if (c.error) RMBLog("GPU command buffer failed: %s", c.error.localizedDescription.UTF8String);
            ReturnToPool(lease);
        }];
        [cb commit];
        if (wait) [cb waitUntilCompleted];
        return cb.error == nil;
    }
}

id<MTLTexture> RMBLegacyFlowForTest(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBFrame& cur,
                                    const RMBFrame& other, RMBSettings s, const RMBGuides* g)
{
    Ctx* ctx = ContextFor(device);
    if (!ctx) return nil;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    LegacyEncoder le = { device, *ctx, [cb computeCommandEncoder], cb };
    id<MTLTexture> flow = le.flow(cur, other, s, g ? &g->prev : nullptr, g ? g->imageTrack : true);
    [le.enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return flow;
}

#ifdef RMB_PROFILE
// Renders once with per-stage GPU timing printed to stdout.
void RMBProfileRender(id<MTLDevice> device, id<MTLCommandQueue> queue, const RMBInputs& in, const RMBFrame& dst,
                      RMBSettings s)
{
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    for (id<MTLCounterSet> cs in device.counterSets)
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) d.counterSet = cs;
    d.sampleCount = 256;
    d.storageMode = MTLStorageModeShared;
    Prof p = { [device newCounterSampleBufferWithDescriptor:d error:nil] };
    gProf = &p;
    MTLTimestamp c0, g0, c1, g1;
    [device sampleTimestamps:&c0 gpuTimestamp:&g0];
    RMBRender(device, queue, in, dst, s, true);
    [device sampleTimestamps:&c1 gpuTimestamp:&g1];
    gProf = nullptr;
    double nsPerTick = double(c1 - c0) / double(g1 - g0);
    NSData* r = [p.buf resolveCounterRange:NSMakeRange(0, 2 * p.names.size())];
    const MTLCounterResultTimestamp* t = (const MTLCounterResultTimestamp*)r.bytes;
    std::map<std::string, double> total;
    std::vector<std::string> order;
    for (size_t i = 0; i < p.names.size(); i++) {
        if (!total.count(p.names[i])) order.push_back(p.names[i]);
        total[p.names[i]] += (t[2 * i + 1].timestamp - t[2 * i].timestamp) * nsPerTick / 1e6;
    }
    for (auto& n : order) printf("    %-28s %7.2f ms\n", n.c_str(), total[n]);
}
#endif

// Legacy mode: pyramid Lucas-Kanade optical flow with block-match propagation, plus the matching gather blur.
// Samplers use pixel coordinates (pixel centres at +0.5).
#include <metal_stdlib>
using namespace metal;

constexpr sampler bs(coord::pixel, filter::linear, address::clamp_to_edge);  // bilinear
constexpr sampler ns(coord::pixel, filter::nearest, address::clamp_to_edge); // nearest

struct LFrame { uint pitch; uint width; uint height; };

// Tracking luma in 0..255, optionally with alpha (Track RGB+A), of the premultiplied frame: AE's straight colour
// under (near-)transparent pixels is arbitrary (even non-finite) and must not look like an edge to the tracker.
// For layer tracking the source is first multiplied by the layer matte (all four channels).
kernel void legacyLuma(device const float4* src [[buffer(0)]],
                       constant LFrame& f        [[buffer(1)]],
                       constant int& addAlpha    [[buffer(2)]],
                       constant int& useMatte    [[buffer(3)]],
                       texture2d<float, access::write> out [[texture(0)]],
                       texture2d<float> matte    [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    float4 p = src[gid.y * f.pitch + gid.x];  // B,G,R,A
    float l = (p.z * 0.3f + p.y * 0.59f + p.x * 0.11f + (addAlpha ? p.w * 0.3f : 0.0f)) * 255.0f;
    if (!(p.w >= 1.0f) || !isfinite(l)) {  // not opaque (kept as a separate path so opaque luma stays bit-exact)
        p = select(p, float4(0), !isfinite(p));
        p.xyz *= saturate(p.w);
        l = (p.z * 0.3f + p.y * 0.59f + p.x * 0.11f + (addAlpha ? p.w * 0.3f : 0.0f)) * 255.0f;
    }
    if (useMatte) l *= matte.read(gid).x;  // = luma of (frame * matte); applied after so plain tracking is unchanged
    out.write(l, gid);
}

// Separable filter in one pass (taps that fall off the image are dropped and the remaining weights renormalised;
// w[0..2r] holds the taps for offsets -r..r): a 16x16 output tile loads its (16 + 2*7)^2 neighbourhood into threadgroup
// memory, filters horizontally into a second tile, then vertically. Same taps, same order, same renormalisation.
constant constexpr int kTile = 16, kMaxR = 7, kSpan = kTile + 2 * kMaxR;
kernel void legacyFilter2D(texture2d<float> in                  [[texture(0)]],
                           texture2d<float, access::write> out   [[texture(1)]],
                           constant float* w                     [[buffer(0)]],
                           constant int& r                       [[buffer(1)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint2 lt [[thread_position_in_threadgroup]])
{
    threadgroup float4 src[kSpan][kSpan];
    threadgroup float4 hor[kSpan][kTile];
    int W = in.get_width(), H = in.get_height();
    int2 o = int2(tg) * kTile - kMaxR;
    uint t = lt.y * kTile + lt.x;
    for (uint i = t; i < kSpan * kSpan; i += kTile * kTile) {
        int2 p = o + int2(i % kSpan, i / kSpan);
        src[i / kSpan][i % kSpan] = (p.x >= 0 && p.y >= 0 && p.x < W && p.y < H) ? in.read(uint2(p)) : float4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < kSpan * kTile; i += kTile * kTile) {
        int row = i / kTile, col = i % kTile, x = int(tg.x) * kTile + col;
        float4 sum = 0;
        float wsum = 0;
        for (int k = -r; k <= r; k++) {
            if (x + k < 0 || x + k >= W) continue;
            sum += w[k + r] * src[row][col + kMaxR + k];
            wsum += w[k + r];
        }
        hor[row][col] = wsum > 0 ? sum / wsum : float4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int2 p = int2(tg) * kTile + int2(lt);
    if (p.x >= W || p.y >= H) return;
    float4 sum = 0;
    float wsum = 0;
    for (int k = -r; k <= r; k++) {
        if (p.y + k < 0 || p.y + k >= H) continue;
        sum += w[k + r] * hor[lt.y + kMaxR + k][lt.x];
        wsum += w[k + r];
    }
    out.write(sum / wsum, uint2(p));
}

// Corner-aligned bilinear resample, result scaled per channel (vectors scale with size).
kernel void legacyResample(texture2d<float> in                [[texture(0)]],
                           texture2d<float, access::write> out [[texture(1)]],
                           constant float2& is                 [[buffer(0)]],  // (srcW-1)/(dstW-1), (srcH-1)/(dstH-1)
                           constant float4& scale              [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    out.write(scale * in.sample(bs, float2(gid) * is + 0.5f), gid);
}

// Central-difference gradient (one-sided at borders) and A = (gx^2, gxgy, gy^2).
// Coverage (layer tracking): g = 0 if any 4-neighbour of cov1 is < 0.95. Plain tracking binds an all-ones cov1.
kernel void legacyGradAndA(texture2d<float> img                     [[texture(0)]],
                           texture2d<float, access::write> grad     [[texture(1)]],
                           texture2d<float, access::write> A        [[texture(2)]],
                           texture2d<float> cov1                    [[texture(3)]],
                           constant int& useCov                     [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]])
{
    uint W = img.get_width(), H = img.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float c = img.read(gid).x;
    float2 g = 0, n = 0;
    if (gid.x > 0)     { g.x += c - img.read(gid - uint2(1, 0)).x; n.x += 1; }
    if (gid.y > 0)     { g.y += c - img.read(gid - uint2(0, 1)).x; n.y += 1; }
    if (gid.x + 1 < W) { g.x += img.read(gid + uint2(1, 0)).x - c; n.x += 1; }
    if (gid.y + 1 < H) { g.y += img.read(gid + uint2(0, 1)).x - c; n.y += 1; }
    g /= n;
    float2 pc = float2(gid) + 0.5f;
    if (useCov && (cov1.sample(ns, pc + float2(0, -1)).x < 0.95f || cov1.sample(ns, pc + float2(0, 1)).x < 0.95f ||
        cov1.sample(ns, pc + float2(-1, 0)).x < 0.95f || cov1.sample(ns, pc + float2(1, 0)).x < 0.95f))
        g = 0;
    grad.write(float4(g, 0, 0), gid);
    A.write(float4(g.x * g.x, g.x * g.y, g.y * g.y, 1), gid);
}

// b = (I0(x - d) - I1(x)) * grad, zero where the warp leaves the image.
kernel void legacyCalcB(texture2d<float> I1                  [[texture(0)]],
                        texture2d<float> I0                  [[texture(1)]],
                        texture2d<float> grad                [[texture(2)]],
                        texture2d<float> disp                [[texture(3)]],
                        texture2d<float, access::write> b    [[texture(4)]],
                        texture2d<float> cov0                [[texture(5)]],
                        constant int& useCov                 [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]])
{
    uint W = I1.get_width(), H = I1.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 c = float2(gid) + 0.5f;
    float2 d = disp.sample(bs, c).xy;
    float2 g = grad.sample(bs, c).xy;
    float2 at = float2(gid) - d;
    float delta = I0.sample(bs, at + 0.5f).x - I1.sample(bs, c).x;
    bool out = (useCov && cov0.sample(bs, at + 0.5f).x < 0.999f) || at.x < 0 || at.x > W - 1 || at.y < 0 || at.y > H - 1;
    b.write(out ? float4(0) : float4(delta * g, 0, 0), gid);
}

// Solve the 2x2 windowed system, skipping near-singular A (|det| <= epsilon).
kernel void legacyNewDisp(texture2d<float> dispIn                  [[texture(0)]],
                          texture2d<float> A                       [[texture(1)]],
                          texture2d<float> b                       [[texture(2)]],
                          texture2d<float, access::write> dispOut  [[texture(3)]],
                          constant float& epsilon                  [[buffer(0)]],
                          constant float& maxStep                  [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= A.get_width() || gid.y >= A.get_height()) return;
    float4 a = A.read(gid);
    float2 B = b.read(gid).xy;
    float4 d = dispIn.read(gid);
    float t = a.y * a.y - a.x * a.z;
    if (fabs(t) > epsilon) {
        float2 u = float2(a.y * B.y - a.z * B.x, a.y * B.x - a.x * B.y) / t;
        float l = length(u);
        if (l > maxStep) u *= maxStep / l;  // a near-singular solve (flat area by a strong edge) must not fling d away
        d.xy += u;
    }
    dispOut.write(d, gid);
}

// 3x3 SSD of the source block against `to` around (x2, y2), bilinear.
// The top-left term is counted twice; kept for the Legacy look.
// The target block is rejected if any of its (bilinear) coverage samples is < 0.95.
static bool matchBlock(texture2d<float> to, texture2d<float> cov0, bool useCov, thread const float s[9], float2 p,
                       uint W, uint H, thread float& r)
{
    if (p.x > W - 2.0f || p.y > H - 2.0f || p.x < 1.0f || p.y < 1.0f) return false;
    if (useCov)
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            if (cov0.sample(bs, p + float2(i - 1, j - 1) + 0.5f).x < 0.95f) return false;
    float e[9];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            e[j * 3 + i] = to.sample(bs, p + float2(i - 1, j - 1) + 0.5f).x - s[j * 3 + i];
    r = e[0] * e[0];
    for (int k = 0; k < 9; k++) r += e[k] * e[k];
    return true;
}

static bool sourceCovered(texture2d<float> cov1, int2 c)
{
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++)
            if (cov1.sample(ns, float2(c + int2(i, j)) + 0.5f).x < 0.95f) return false;
    return true;
}

static void sourceBlock(texture2d<float> from, int2 c, thread float s[9])
{
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            s[j * 3 + i] = from.sample(ns, float2(c + int2(i - 1, j - 1)) + 0.5f).x;  // clamp to edge
}

// Parallel pre-pass for one sweep direction. own = match cost of the pixel's own vector; pred = cost of its sweep
// predecessor's (input) vector, which is exactly what the serial sweep needs whenever the predecessor kept its
// own vector (the common case). -1 marks "block leaves the image".
kernel void legacySweepCosts(texture2d<float> from                  [[texture(0)]],  // I1
                             texture2d<float> to                    [[texture(1)]],  // I0
                             texture2d<float> flowIn                [[texture(2)]],
                             texture2d<float, access::write> costs  [[texture(3)]],  // rg: own, pred
                             texture2d<float> cov1                  [[texture(4)]],
                             texture2d<float> cov0                  [[texture(5)]],
                             constant int& dir                      [[buffer(0)]],
                             constant int& useCov                   [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]])
{
    uint W = from.get_width(), H = from.get_height();
    if (gid.x >= W || gid.y >= H) return;
    int2 back = dir == 0 ? int2(-1, 0) : dir == 1 ? int2(1, 0) : dir == 2 ? int2(0, -1) : int2(0, 1);
    int2 q = clamp(int2(gid) + back, int2(0), int2(W - 1, H - 1));
    float s[9], co, cp;
    sourceBlock(from, int2(gid), s);
    // An uncovered source block always keeps its own vector (own cost marked invalid).
    bool okOwn = (!useCov || sourceCovered(cov1, int2(gid))) &&
                 matchBlock(to, cov0, useCov, s, float2(gid) - flowIn.read(gid).xy, W, H, co);
    bool okPred = matchBlock(to, cov0, useCov, s, float2(gid) - flowIn.read(uint2(q)).xy, W, H, cp);
    costs.write(float4(okOwn ? co : -1.0f, okPred ? cp : -1.0f, 0, 0), gid);
}

// Serial propagation along one scanline/column. Each pixel keeps its own vector or the one just chosen for its
// predecessor, whichever matches better. Border pixels are copied. Precomputed costs (legacySweepCosts) cover
// every step except those following a propagated vector, which are evaluated here.
// dir: 0 = rows L->R, 1 = rows R->L, 2 = columns T->B, 3 = columns B->T. One thread per row/column.
// Note: the serial chain is the tracker's main cost (~1 us/step); batching and run speculation measured slower
kernel void legacyBlockMatch(texture2d<float> from                     [[texture(0)]],  // I1
                             texture2d<float> to                       [[texture(1)]],  // I0
                             texture2d<float> flowIn                   [[texture(2)]],
                             texture2d<float, access::write> flowOut   [[texture(3)]],
                             texture2d<float> costs                    [[texture(4)]],
                             texture2d<float> cov0                     [[texture(5)]],
                             constant int& dir                         [[buffer(0)]],
                             constant int& useCov                      [[buffer(1)]],
                             uint tid [[thread_position_in_grid]])
{
    uint W = from.get_width(), H = from.get_height();
    bool rows = dir < 2, fwd = (dir & 1) == 0;
    uint len = rows ? W : H, lane = tid + 1;
    if (lane >= (rows ? H : W) - 1) return;
    int step = fwd ? 1 : -1;

    uint first = fwd ? 0 : len - 1, last = fwd ? len - 1 : 0;
    uint2 p0 = rows ? uint2(first, lane) : uint2(lane, first);
    float4 prev = flowIn.read(p0);
    flowOut.write(prev, p0);
    bool propagated = false;  // prev is a carried vector, not the predecessor's own input vector
    for (uint k = first + step; k != last; k += step) {
        uint2 p = rows ? uint2(k, lane) : uint2(lane, k);
        float2 c = costs.read(p).xy;
        float cPrev = c.y;
        bool okPrev = c.y >= 0;
        if (propagated && c.x >= 0) {
            float s[9];
            sourceBlock(from, int2(p), s);
            okPrev = matchBlock(to, cov0, useCov, s, float2(p) - prev.xy, W, H, cPrev);
        }
        propagated = c.x >= 0 && okPrev && cPrev < c.x;
        float4 pick = propagated ? prev : flowIn.read(p);
        flowOut.write(pick, p);
        prev = pick;
    }
    uint2 pl = rows ? uint2(last, lane) : uint2(lane, last);
    flowOut.write(flowIn.read(pl), pl);
}

// Track Enhance "Delinearize": I = 255 * (max(I, 0) / 255)^2.2, applied once at tracking resolution.
kernel void legacyDelinearize(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) return;
    out.write(255.0f * pow(max(in.read(gid).x, 0.0f) / 255.0f, 2.2f), gid);
}

kernel void legacyClear(texture2d<float, access::write> t [[texture(0)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height()) t.write(float4(0), gid);
}

struct LegacyBlur {
    float2 ratio;   // flow size / image size
    float2 scale;   // 0.5 / ratio * amount
    float mix;      // 1 = blur; < 0 = negative amount: src + (blur - src) * mix
    int hq;         // bilinear (AE Best) vs nearest (AE Draft)
};

static float4 bilinearBuf(device const float4* img, uint pitch, uint W, uint H, float2 p)
{
    p = clamp(p, float2(0), float2(W - 1, H - 1));  // clamp to edge
    uint2 i = uint2(p);
    uint2 j = min(i + 1, uint2(W - 1, H - 1));
    float2 t = p - float2(i);
    return mix(mix(img[i.y * pitch + i.x], img[i.y * pitch + j.x], t.x),
               mix(img[j.y * pitch + i.x], img[j.y * pitch + j.x], t.x), t.y);
}

// Average samples along +/- the pixel's own vector (centred),
// skipping samples that leave the frame.
kernel void legacyMotionBlur(device const float4* src [[buffer(0)]],
                             device float4* dst       [[buffer(1)]],
                             constant LFrame& sf      [[buffer(2)]],
                             constant LFrame& df      [[buffer(3)]],
                             constant LegacyBlur& p   [[buffer(4)]],
                             texture2d<float> flow    [[texture(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= df.width || gid.y >= df.height) return;
    float4 center = src[min(gid.y, sf.height - 1) * sf.pitch + min(gid.x, sf.width - 1)];
    float2 v = p.scale * flow.sample(bs, float2(gid) * p.ratio + 0.5f).xy;
    if (all(v == 0) || !all(isfinite(v))) { dst[gid.y * df.pitch + gid.x] = center; return; }  // exact copy
    int idist = int(max(length(v) + 1.0f, 2.0f));
    float2 delta = v / float(idist - 1);
    float4 sum = 0;
    int used = 0;
    for (int l = -(idist - 1); l < idist; l++) {
        float2 c = float2(gid) + delta * float(l);
        if (p.hq) {
            if (c.x < 0 || c.y < 0 || c.x >= sf.width || c.y >= sf.height) continue;
            sum += bilinearBuf(src, sf.pitch, sf.width, sf.height, c);
        } else {
            int2 t = int2(c + 0.5f);  // round via (int)(f + 0.5)
            if (t.x < 0 || t.y < 0 || t.x >= int(sf.width) || t.y >= int(sf.height)) continue;
            sum += src[t.y * sf.pitch + t.x];
        }
        used++;
    }
    float4 blur = sum / float(used);
    dst[gid.y * df.pitch + gid.x] = p.mix == 1.0f ? blur : center + (blur - center) * p.mix;
}

// While still checking, if no vector at this level lands inside the image, zero the field;
// once any level has an in-bounds vector, stop checking. state[0] = still checking, state[1] = any inside (scratch).
kernel void legacyAnyInside(texture2d<float> disp      [[texture(0)]],
                            device atomic_uint* state  [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    uint W = disp.get_width(), H = disp.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 p = float2(gid) + disp.read(gid).xy;
    bool inside = p.x >= 0 && p.x <= W - 1 && p.y >= 0 && p.y <= H - 1;
    if (simd_any(inside) && simd_is_first()) atomic_store_explicit(&state[1], 1u, memory_order_relaxed);
}

kernel void legacyZeroIfAllOutside(texture2d<float, access::read_write> disp [[texture(0)]],
                                   device const uint* state                   [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= disp.get_width() || gid.y >= disp.get_height()) return;
    if (state[0] && !state[1]) disp.write(float4(0), gid);
}

kernel void legacyUpdateCheck(device uint* state [[buffer(0)]], uint tid [[thread_position_in_grid]])
{
    if (state[1]) state[0] = 0;
    state[1] = 0;
}

// New mode's replacement for the serial block match at the finest level: every pixel at once keeps whichever of its
// own vector and its four neighbours' at distance `step` matches its 3x3 block best. Run with shrinking steps it
// spreads good vectors as far as the serial sweeps, fully in parallel.
kernel void newPropagate(texture2d<float> from                     [[texture(0)]],  // I1
                         texture2d<float> to                       [[texture(1)]],  // I0
                         texture2d<float> flowIn                   [[texture(2)]],
                         texture2d<float, access::write> flowOut   [[texture(3)]],
                         texture2d<float> cov0                     [[texture(4)]],
                         constant int& step                        [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    uint W = from.get_width(), H = from.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float4 best = flowIn.read(gid);
    float s[9], c, bc;
    sourceBlock(from, int2(gid), s);
    if (!matchBlock(to, cov0, false, s, float2(gid) - best.xy, W, H, bc)) { flowOut.write(best, gid); return; }
    const int2 off[4] = { int2(-step, 0), int2(step, 0), int2(0, -step), int2(0, step) };
    for (int k = 0; k < 4; k++) {
        int2 q = int2(gid) + off[k];
        if (q.x < 0 || q.y < 0 || q.x >= int(W) || q.y >= int(H)) continue;
        float4 v = flowIn.read(uint2(q));
        if (matchBlock(to, cov0, false, s, float2(gid) - v.xy, W, H, c) && c < bc) { bc = c; best = v; }
    }
    flowOut.write(best, gid);
}

// New mode's first pyramid step: tracking luma (as legacyLuma, premultiplied) straight at half resolution, filtered
// by a centre-aligned 6x6 binomial ([1 5 10 10 5 1]/32 per axis, edge taps renormalised) of the full-resolution pixels. One pass
// instead of luma + filter + resample at full size.
kernel void newLumaHalf(device const float4* src [[buffer(0)]],
                        constant LFrame& f        [[buffer(1)]],
                        constant int& addAlpha    [[buffer(2)]],
                        texture2d<float, access::write> out [[texture(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    const float w[6] = { 1, 5, 10, 10, 5, 1 };
    float sum = 0, wsum = 0;
    for (int j = 0; j < 6; j++)
        for (int i = 0; i < 6; i++) {
            int2 p = int2(gid) * 2 + int2(i - 2, j - 2);
            if (p.x < 0 || p.y < 0 || p.x >= int(f.width) || p.y >= int(f.height)) continue;
            float4 v = src[p.y * f.pitch + p.x];
            v = select(v, float4(0), !isfinite(v));
            v.xyz *= saturate(v.w);
            float wt = w[i] * w[j];
            sum += wt * (v.z * 0.3f + v.y * 0.59f + v.x * 0.11f + (addAlpha ? v.w * 0.3f : 0.0f)) * 255.0f;
            wsum += wt;
        }
    out.write(sum / wsum, gid);
}

// Layered (foreground matte) processing for Legacy mode: matte extraction and choke, layer split, warp from the
// tracking neighbour, push-pull hole fill and premultiplied compositing. Layer images are rgba32Float textures,
// premultiplied, channel order as AE's GPU frames (B, G, R, A).
#include <metal_stdlib>
using namespace metal;

constexpr sampler bsL(coord::pixel, filter::linear, address::clamp_to_edge);

struct MFrame { uint pitch; uint width; uint height; };

// One channel of the matte layer, clamped to 0..1. chan follows the popup: 0 Red, 1 Green, 2 Blue, 3 Alpha.
kernel void layerMatteChannel(device const float4* src [[buffer(0)]],
                              constant MFrame& f        [[buffer(1)]],
                              constant int& chan        [[buffer(2)]],
                              texture2d<float, access::write> out [[texture(0)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    float4 p = src[gid.y * f.pitch + gid.x];
    float m = chan == 0 ? p.z : chan == 1 ? p.y : chan == 2 ? p.x : p.w;
    out.write(isfinite(m) ? saturate(m) : 0.0f, gid);  // float mattes can overshoot
}

// Choke helpers. Invert for growing: 1 - x, snapped to 1 above 0.999.
kernel void layerChokeInvert(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) return;
    float v = 1.0f - in.read(gid).x;
    out.write(v > 1.0f - 1e-3f ? 1.0f : v, gid);
}

// Threshold a blurred matte: below mn -> 0, else rescale to 0..1; within 1e-4 of 1 -> 1.
kernel void layerChokeMap(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                          constant float& mn [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) return;
    float v = in.read(gid).x;
    v = v < mn ? 0.0f : (v - mn) / (1.0f - mn);
    out.write(fabs(v - 1.0f) < 1e-4f ? 1.0f : v, gid);
}

// a + (b - a) * f, clamped to 0..1 when clampOut is set.
kernel void layerBlend(texture2d<float> a [[texture(0)]], texture2d<float> b [[texture(1)]],
                       texture2d<float, access::write> out [[texture(2)]], constant float& f [[buffer(0)]],
                       constant int& clampOut [[buffer(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float4 v = a.read(gid) + (b.read(gid) - a.read(gid)) * f;
    out.write(clampOut ? saturate(v) : v, gid);
}

// Layer matte from a raw matte and the (grown) mattes in front of it: clamp((1 - front) * raw).
kernel void layerBehindMatte(texture2d<float> raw [[texture(0)]], texture2d<float> front [[texture(1)]],
                             texture2d<float, access::write> out [[texture(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    out.write(saturate((1.0f - front.read(gid).x) * raw.read(gid).x), gid);
}

// Coverage for tracking is binary: >= 0.95 counts as covered.
kernel void layerBinarize(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) return;
    out.write(in.read(gid).x < 0.95f ? 0.0f : 1.0f, gid);
}

// Layer colour: source frame times the layer matte, all four channels (premultiplied stays premultiplied).
kernel void layerColor(device const float4* src [[buffer(0)]], constant MFrame& f [[buffer(1)]],
                       texture2d<float> matte [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    out.write(src[gid.y * f.pitch + gid.x] * matte.read(gid).x, gid);
}

// Warp the neighbour's layer into the current frame: W(x) = Cn(x - d), bilinear. d is the tracker's displacement
// (current(x) ~ neighbour(x - d)) at flow resolution; a sample point outside the frame is pulled back onto the
// edge along the ray from x.
kernel void layerWarp(texture2d<float> cn [[texture(0)]], texture2d<float> flow [[texture(1)]],
                      texture2d<float, access::write> out [[texture(2)]], constant float2& ratio [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]])
{
    uint W = out.get_width(), H = out.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 d = flow.sample(bsL, float2(gid) * ratio + 0.5f).xy / ratio;
    float x = gid.x, y = gid.y, ix = x - d.x, iy = y - d.y, w1 = W - 1.0f, h1 = H - 1.0f;
    if (ix < 0)  { float ny = y - ((x - 0) / (x - ix)) * (y - iy);  ix = 0;  iy = ny; }
    if (ix > w1) { float ny = y - ((x - w1) / (x - ix)) * (y - iy); ix = w1; iy = ny; }
    if (iy < 0)  { float nx = x - ((y - 0) / (y - iy)) * (x - ix);  iy = 0;  ix = nx; }
    if (iy > h1) { float nx = x - ((y - h1) / (y - iy)) * (x - ix); iy = h1; ix = nx; }
    out.write(cn.sample(bsL, float2(ix, iy) + 0.5f), gid);
}

// Premultiplied over: rgb = dst * (1 - a_src) + src, a = a_dst + (1 - a_dst) * a_src.
kernel void layerOver(texture2d<float> dst [[texture(0)]], texture2d<float> src [[texture(1)]],
                      texture2d<float, access::write> out [[texture(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float4 d = dst.read(gid), s = src.read(gid);
    out.write(float4(d.xyz * (1.0f - s.w) + s.xyz, d.w + (1.0f - d.w) * s.w), gid);
}

// Hole-fill setup on the unpremultiplied layer: a pixel is valid when its alpha is (almost) 1.
// B = colour * valid, C = valid.
kernel void layerFillSeed(texture2d<float> premult [[texture(0)]], texture2d<float, access::write> B [[texture(1)]],
                          texture2d<float, access::write> C [[texture(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= premult.get_width() || gid.y >= premult.get_height()) return;
    float4 v = premult.read(gid);
    if (v.w >= 1e-7f) v.xyz /= v.w;  // unpremultiply
    bool valid = 1.0f - v.w < 0.000031f;
    B.write(valid ? v : float4(0), gid);
    C.write(valid ? 1.0f : 0.0f, gid);
}

// Push-pull filters: binomial [1 4 6 4 1]/16 horizontally with mirrored edges ...
kernel void layerBinomialFlipH(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                               uint2 gid [[thread_position_in_grid]])
{
    int W = in.get_width();
    if (int(gid.x) >= W || gid.y >= in.get_height()) return;
    int w1 = W - 1, x = gid.x;
    const float wt[5] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };
    float4 sum = 0.375f * in.read(gid);
    float denom = 0.375f;
    for (int k = -2; k <= 2; k++) {
        if (k == 0) continue;
        int xi = x + k;
        xi = k < 0 ? (xi >= 0 ? xi : -xi) : (xi > w1 ? 2 * w1 - xi : xi);
        if (xi < 0 || xi > w1) continue;
        sum += wt[k + 2] * in.read(uint2(xi, gid.y));
        denom += wt[k + 2];
    }
    out.write(sum / denom, gid);
}

// ... and vertically with renormalised edges.
kernel void layerBinomialV(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]])
{
    int H = in.get_height();
    if (gid.x >= in.get_width() || int(gid.y) >= H) return;
    const float wt[5] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };
    float4 sum = 0;
    float denom = 0;
    for (int k = -2; k <= 2; k++) {
        int yi = int(gid.y) + k;
        if (yi < 0 || yi >= H) continue;
        sum += wt[k + 2] * in.read(uint2(gid.x, yi));
        denom += wt[k + 2];
    }
    out.write(sum / denom, gid);
}

// Pull step: where the level has valid samples, A = B / C; elsewhere keep the value coming from coarser levels.
kernel void layerFillAdjust(texture2d<float> A [[texture(0)]], texture2d<float> B [[texture(1)]],
                            texture2d<float> C [[texture(2)]], texture2d<float, access::write> out [[texture(3)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float c = C.read(gid).x;
    out.write(c > 0 ? B.read(gid) / c : A.read(gid), gid);
}

// Final layer alpha (FG: the raw matte; BG: the source's alpha) and premultiply.
kernel void layerFinish(texture2d<float> filled [[texture(0)]], texture2d<float> alpha [[texture(1)]],
                        device const float4* src [[buffer(0)]], constant MFrame& f [[buffer(1)]],
                        constant int& alphaFromSrc [[buffer(2)]], texture2d<float, access::write> out [[texture(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float a = alphaFromSrc ? src[gid.y * f.pitch + gid.x].w : alpha.read(gid).x;
    float4 v = filled.read(gid);
    out.write(float4(v.xyz * a, a), gid);
}

// Texture <-> AE-layout buffer copies around the gather blur, which works on buffers.
kernel void layerToBuffer(texture2d<float> in [[texture(0)]], device float4* dst [[buffer(0)]],
                          constant MFrame& f [[buffer(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    dst[gid.y * f.pitch + gid.x] = in.read(gid);
}

// Back-to-front premultiplied composite of blurred layers into the output frame.
kernel void layerCompositeOut(device const float4* bg [[buffer(0)]], device const float4* fg [[buffer(1)]],
                              device float4* dst [[buffer(2)]], constant MFrame& lf [[buffer(3)]],
                              constant MFrame& df [[buffer(4)]], constant int& mode [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= df.width || gid.y >= df.height) return;
    uint i = gid.y * lf.pitch + gid.x;
    float4 d = bg[i], s = fg[i];
    float4 v = mode == 2 ? s : mode == 3 ? d : float4(d.xyz * (1.0f - s.w) + s.xyz, d.w + (1.0f - d.w) * s.w);
    dst[gid.y * df.pitch + gid.x] = v;
}

static float visibleLuma(float4 p)  // luma of what is visible (as the tracker sees it), 0.29R + 0.59G + 0.12B
{
    p = select(p, float4(0), !isfinite(p));
    return dot(p.xyz * saturate(p.w), float3(0.12f, 0.59f, 0.29f));  // B, G, R
}

// Mean absolute luma difference (0..1) of frame a against b and against c, accumulated into sum[0] and sum[1]
// (one pass over a for both neighbours).
kernel void frameDiff(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                      device const float4* c [[buffer(2)]], constant MFrame& fa [[buffer(3)]],
                      constant MFrame& fb [[buffer(4)]], constant MFrame& fc [[buffer(5)]],
                      device atomic_float* sum [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    float vb = 0, vc = 0;
    if (gid.x < fa.width && gid.y < fa.height) {
        float l = visibleLuma(a[gid.y * fa.pitch + gid.x]);
        vb = fabs(l - visibleLuma(b[gid.y * fb.pitch + gid.x]));
        vc = fabs(l - visibleLuma(c[gid.y * fc.pitch + gid.x]));
    }
    vb = simd_sum(vb);
    vc = simd_sum(vc);
    if (simd_is_first()) {
        atomic_fetch_add_explicit(&sum[0], vb, memory_order_relaxed);
        atomic_fetch_add_explicit(&sum[1], vc, memory_order_relaxed);
    }
}

kernel void layerSetValue(texture2d<float, access::write> t [[texture(0)]], constant float& v [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < t.get_width() && gid.y < t.get_height()) t.write(float4(v), gid);
}

// One pass of a separable filter of any radius (the matte choke needs radii above the 2D kernel's limit); taps
// that fall off the image are dropped and the rest renormalised.
kernel void layerSepFilter(texture2d<float> in [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                           constant float* w [[buffer(0)]], constant int& r [[buffer(1)]],
                           constant int& vertical [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    int W = in.get_width(), H = in.get_height();
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    float4 sum = 0;
    float wsum = 0;
    for (int i = -r; i <= r; i++) {
        int2 p = int2(gid) + (vertical ? int2(0, i) : int2(i, 0));
        if (p.x < 0 || p.y < 0 || p.x >= W || p.y >= H) continue;
        sum += w[i + r] * in.read(uint2(p));
        wsum += w[i + r];
    }
    out.write(sum / wsum, gid);
}

// AE passes straight alpha; blurring and compositing work premultiplied, and the result goes back straight.
kernel void premultiply(device const float4* src [[buffer(0)]], constant MFrame& sf [[buffer(1)]],
                        device float4* dst [[buffer(2)]], constant MFrame& df [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= sf.width || gid.y >= sf.height) return;
    float4 v = src[gid.y * sf.pitch + gid.x];
    v = select(v, float4(0), !isfinite(v));  // AE can hand over non-finite colour under transparency
    dst[gid.y * df.pitch + gid.x] = float4(v.xyz * v.w, v.w);
}

kernel void unpremultiply(device float4* buf [[buffer(0)]], constant MFrame& f [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    float4 v = buf[gid.y * f.pitch + gid.x];
    if (v.w > 0 && v.w != 1) buf[gid.y * f.pitch + gid.x] = float4(v.xyz / v.w, v.w);
}

// Guides (track points, mask samples) as a sparse image on a cleared texture: entry = (pixel x, pixel y, d.x, d.y)
// becomes (d.x, d.y, 1, 0). The entries' pixels are unique.
kernel void guideScatter(device const float4* e [[buffer(0)]], constant uint& n [[buffer(1)]],
                         texture2d<float, access::write> out [[texture(0)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) out.write(float4(e[gid].zw, 1, 0), uint2(e[gid].xy));
}

// Where the guide image's weight (z) is above thr, the displacement becomes its weighted mean (xy / z); elsewhere
// it is kept.
kernel void guideAdjust(texture2d<float> in [[texture(0)]], texture2d<float> g [[texture(1)]],
                        texture2d<float, access::write> out [[texture(2)]], constant float& thr [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float4 c = g.read(gid);
    out.write(c.z > thr ? float4(c.xy / c.z, 0, 0) : in.read(gid), gid);
}

// sRGB encoding (inverse of the sRGB decode a linearised working space applies).
static float srgbEncode(float c)
{
    c = max(c, 0.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
}

// Vector Input decode: red -> x, green -> y (+y up, so negated for rows), 0.5 = no motion. A vector is valid where
// the vector layer is opaque; B = vector * valid, C = valid feed the push-pull fill. linear: the layer arrived
// linearised, so its values are sRGB-encoded back first.
kernel void vectorSeed(device const float4* v [[buffer(0)]], constant MFrame& f [[buffer(1)]],
                       constant float& maxD [[buffer(2)]], constant int& linear [[buffer(3)]],
                       texture2d<float, access::write> B [[texture(0)]],
                       texture2d<float, access::write> C [[texture(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    float4 p = v[gid.y * f.pitch + gid.x];  // B, G, R, A
    if (linear) p.yz = float2(srgbEncode(p.y), srgbEncode(p.z));
    bool valid = 1.0f - p.w < 0.000031f;
    B.write(valid ? float4(maxD * (2 * p.z - 1), -maxD * (2 * p.y - 1), 0, 0) : float4(0), gid);
    C.write(valid ? 1.0f : 0.0f, gid);
}

// Last pass of every render: non-finite output values (a broken input or a numeric corner case) become 0, counted
// into n[0], so they never reach AE.
kernel void finiteGuard(device float4* buf [[buffer(0)]], constant MFrame& f [[buffer(1)]],
                        device atomic_uint* n [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= f.width || gid.y >= f.height) return;
    float4 v = buf[gid.y * f.pitch + gid.x];
    if (all(isfinite(v))) return;
    buf[gid.y * f.pitch + gid.x] = select(v, float4(0), !isfinite(v));
    atomic_fetch_add_explicit(n, 1, memory_order_relaxed);
}

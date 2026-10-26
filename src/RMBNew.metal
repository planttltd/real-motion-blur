// New mode: depth-bucketed reconstruction blur, fed by the Legacy tracker run in both directions.
// Displacements are anchored on the current frame: cur(x) ~ prev(x - dPrev) ~ next(x - dNext).
#include <metal_stdlib>
using namespace metal;

constexpr sampler normLinear(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler pixLinear(coord::pixel, filter::linear, address::clamp_to_edge);

struct NFrame { uint pitch; uint width; uint height; };

// ---- Reconstruction blur ----------------------------------------------------------------------------------------

struct NBlur {
    uint w, h;      // image size
    float scale;    // blur amount * 0.5: half streak length per unit of per-frame velocity
    float mix;      // 1 = blur; < 0 = negative amount: src + (blur - src) * mix
    int mode;       // velocity at t: 0 = (dPrev - dNext)/2, 1 = next only (-dNext), 2 = previous only (dPrev)
    int tile;       // tile size in pixels
    int samples;    // minimum front-layer samples
    float softZ;    // depth softness
    float ownDensity;  // own-layer samples per pixel of half streak
    int matteDepth;    // depth: 1 = the FG matte, 0 = the layer's alpha
};

// Half-streak velocity (image pixels) at pixel-centre position X, read straight from the tracker's fields
// (tracking resolution, bilinear).
static float2 velAt(texture2d<float> dPrev, texture2d<float> dNext, constant NBlur& b, float2 X)
{
    float2 uv = X / float2(b.w, b.h);
    texture2d<float> ref = b.mode == 1 ? dNext : dPrev;
    float2 toImg = float2(b.w, b.h) / float2(ref.get_width(), ref.get_height());
    float2 p = dPrev.sample(normLinear, uv).xy, n = dNext.sample(normLinear, uv).xy;
    float2 v = b.mode == 0 ? 0.5 * (p - n) : b.mode == 1 ? -n : p;
    v *= toImg * b.scale;
    return all(isfinite(v)) ? v : 0;
}

// Depth (larger = nearer): the FG matte, else the layer's alpha.
static float depthAt(device const float4* src, constant NFrame& sf, texture2d<float> depthTex, constant NBlur& b,
                     uint2 p)
{
    return b.matteDepth ? depthTex.read(p).x : src[p.y * sf.pitch + p.x].w;
}

// Per tile: the longest velocity of any visible pixel (an empty, fully transparent pixel has nothing to spread,
// and the tracker's guesses there must not steer the front pass), and the depth range: (v.x, v.y, zmin, zmax).
kernel void newTileMax(device float4* tmax        [[buffer(1)]],
                       constant NBlur& b          [[buffer(2)]],
                       device const float4* src   [[buffer(4)]],
                       constant NFrame& sf        [[buffer(5)]],
                       texture2d<float> dPrev     [[texture(0)]],
                       texture2d<float> dNext     [[texture(1)]],
                       texture2d<float> depthTex  [[texture(2)]],
                       uint2 tile [[threadgroup_position_in_grid]],
                       uint2 lt [[thread_position_in_threadgroup]],
                       uint li [[thread_index_in_threadgroup]],
                       uint sl [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]])
{
    // One 32x8 threadgroup per tile; each thread scans a column of the tile, then the group reduces.
    uint tw = (b.w + b.tile - 1) / b.tile;
    float2 best = 0;
    float bl = 0, zmin = INFINITY, zmax = -INFINITY;
    uint x = tile.x * b.tile + lt.x;
    for (uint y = tile.y * b.tile + lt.y; x < b.w && y < min(b.h, (tile.y + 1) * b.tile); y += 8) {
        float z = depthAt(src, sf, depthTex, b, uint2(x, y));
        zmin = min(zmin, z);
        zmax = max(zmax, z);
        if (!b.matteDepth && z <= 0) continue;  // depth is the alpha here
        float2 v = velAt(dPrev, dNext, b, float2(x, y) + 0.5);
        float l = dot(v, v);
        if (l > bl) { bl = l; best = v; }
    }
    zmin = simd_min(zmin);
    zmax = simd_max(zmax);
    float m = simd_max(bl);
    uint who = simd_min(bl == m ? sl : 0xffffffffu);  // lane holding the longest
    float2 v = float2(simd_shuffle(best.x, who), simd_shuffle(best.y, who));
    threadgroup float4 part[8];
    threadgroup float lens[8];
    if (sl == 0) { part[sg] = float4(v, zmin, zmax); lens[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (li == 0) {
        float4 r = part[0];
        float rl = lens[0];
        uint n = (32 * 8 + 31) / 32;
        for (uint k = 1; k < n; k++) {
            if (lens[k] > rl) { rl = lens[k]; r.xy = part[k].xy; }
            r.z = min(r.z, part[k].z);
            r.w = max(r.w, part[k].w);
        }
        tmax[tile.y * tw + tile.x] = r;
    }
}

kernel void newNeighborMax(device const float4* tmax [[buffer(0)]],
                           device float4* nmax       [[buffer(1)]],
                           constant NBlur& b         [[buffer(2)]],
                           uint2 tid [[thread_position_in_grid]])
{
    int tw = (b.w + b.tile - 1) / b.tile, th = (b.h + b.tile - 1) / b.tile;
    if (int(tid.x) >= tw || int(tid.y) >= th) return;
    float2 best = 0;
    float bl = 0, zmin = INFINITY, zmax = -INFINITY;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int2 q = int2(tid) + int2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= tw || q.y >= th) continue;
            float4 t = tmax[q.y * tw + q.x];
            zmin = min(zmin, t.z);
            zmax = max(zmax, t.w);
            float2 v = t.xy;
            // diagonal neighbours only count if they point at this tile
            if (dx != 0 && dy != 0 && abs(dot(normalize(v + 1e-6), normalize(float2(dx, dy)))) < 0.7) continue;
            float l = dot(v, v);
            if (l > bl) { bl = l; best = v; }
        }
    nmax[tid.y * tw + tid.x] = float4(best, zmin, zmax);
}

static float4 sampleImg(device const float4* img, constant NFrame& f, float2 p)  // bilinear, pixel-centre coords
{
    p = clamp(p - 0.5, float2(0), float2(f.width - 1, f.height - 1));
    uint2 i = uint2(p), j = min(i + 1, uint2(f.width - 1, f.height - 1));
    float2 t = p - float2(i);
    return mix(mix(img[i.y * f.pitch + i.x], img[i.y * f.pitch + j.x], t.x),
               mix(img[j.y * f.pitch + i.x], img[j.y * f.pitch + j.x], t.x), t.y);
}

static float noise(float2 p) { return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715)))); }

// Centred gather along v (half streak) around X, bilinear, out-of-frame samples skipped.
static float4 gatherAlong(device const float4* src, constant NFrame& sf, constant NBlur& b, float2 X, float2 v,
                          float4 cX)
{
    int n = clamp(int(length(v) * b.ownDensity + 1.0), 1, 256);
    float2 d = v / float(n);
    float4 sum = cX;
    float cnt = 1;
    for (int l = 1; l <= n; l++)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float2 Y = X + d * float(l * sgn);
            if (Y.x < 0 || Y.y < 0 || Y.x >= b.w || Y.y >= b.h) continue;
            sum += sampleImg(src, sf, Y);
            cnt += 1;
        }
    return sum / cnt;
}

// Depth-bucketed reconstruction (depth: the FG matte, or the layer's alpha). Where the neighbourhood has one depth
// (opaque footage, the inside of layers) it is a plain centred gather along the pixel's own vector. Elsewhere the
// pixel's own layer gets that gather (skipping nearer samples), and nearer layers are splat-gathered along the
// neighbourhood-max vector into a coverage fraction: out = front * coverage + own * (1 - coverage).
kernel void newReconstruct(device const float4* src  [[buffer(0)]],
                           device float4* dst        [[buffer(1)]],
                           constant NFrame& sf       [[buffer(2)]],
                           constant NFrame& df       [[buffer(3)]],
                           constant NBlur& b         [[buffer(4)]],
                           device const float4* nmax [[buffer(7)]],
                           texture2d<float> dPrev    [[texture(0)]],
                           texture2d<float> dNext    [[texture(1)]],
                           texture2d<float> depthTex [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= df.width || gid.y >= df.height) return;
    float4 cX = src[gid.y * sf.pitch + gid.x];
    float2 X = float2(gid) + 0.5;
    float2 vX = velAt(dPrev, dNext, b, X);
    float lX = length(vX);
    int tw = (b.w + b.tile - 1) / b.tile;
    float4 nm = nmax[(gid.y / b.tile) * tw + gid.x / b.tile];
    if (nm.w - nm.z < 1e-4) {  // one depth around here (all opaque footage): a centred gather along the own vector
        float4 blur = lX > 0.05 ? gatherAlong(src, sf, b, X, vX, cX) : cX;
        dst[gid.y * df.pitch + gid.x] = b.mix == 1.0 ? blur : cX + (blur - cX) * b.mix;
        return;
    }
    float2 vn = nm.xy;
    float ln = length(vn);
    if (ln < 0.5) { dst[gid.y * df.pitch + gid.x] = cX; return; }
    float zX = depthAt(src, sf, depthTex, b, gid);

    float4 own = cX;
    float wo = 1;
    if (lX > 0.05) {
        int n = clamp(int(lX * b.ownDensity + 1.0), 1, 256);
        float2 d = vX / float(n);
        for (int l = 1; l <= n; l++)
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float2 Y = X + d * float(l * sgn);
                if (Y.x < 0 || Y.y < 0 || Y.x >= b.w || Y.y >= b.h) continue;
                float zY = depthAt(src, sf, depthTex, b, uint2(Y));
                float w = 1.0 - saturate((zY - zX) / b.softZ - 0.5);
                own += w * sampleImg(src, sf, Y);
                wo += w;
            }
    }
    own /= wo;

    int S = clamp(int(ceil(ln)), b.samples, 64);
    float dens = float(S) / (2.0 * ln);
    float j = noise(X) - 0.5;
    float4 F = 0;
    float wf = 0;
    for (int s = 0; s < S; s++) {
        float t = mix(-1.0, 1.0, (float(s) + j + 0.5) / float(S));
        float2 Y = X + vn * t;
        if (Y.x < 0 || Y.y < 0 || Y.x >= b.w || Y.y >= b.h) continue;
        uint2 iy = uint2(Y);
        float fr = saturate((depthAt(src, sf, depthTex, b, iy) - zX) / b.softZ - 0.5);
        if (fr <= 0) continue;
        float lY = max(length(velAt(dPrev, dNext, b, float2(iy) + 0.5)), 0.5);
        float dist = abs(t) * ln;
        float w = fr * (1.0 - smoothstep(lY - 0.5, lY + 0.5, dist)) / (2.0 * lY);
        F += w * sampleImg(src, sf, Y);
        wf += w;
    }
    float cov = saturate(wf / dens);
    float4 blur = (wf > 0 ? F / wf : 0) * cov + own * (1 - cov);
    // An empty pixel (nothing there in the source) where the tracked vector moves with the layer: gathering along it
    // is exact. The front pass is for where it falls well short (tracked motion static there).
    if (!b.matteDepth && cX.w <= 0 && lX > 0.05) {
        float4 g = gatherAlong(src, sf, b, X, vX, cX);
        if (g.w >= 0.5 * blur.w) blur = g;
    }
    dst[gid.y * df.pitch + gid.x] = b.mix == 1.0 ? blur : cX + (blur - cX) * b.mix;
}

// With a FG matte: two layers, each blurred along its own (cleaned, frame-wide) motion field, FG over BG.
// FG = colour * matte; BG = the background plate (colour outside the FG, the FG's hole filled), both premultiplied.
static float2 fieldVel(texture2d<float> f, constant NBlur& b, float2 X)  // field already "per frame" (mode 2)
{
    float2 v = f.sample(normLinear, X / float2(b.w, b.h)).xy;
    v *= float2(b.w, b.h) / float2(f.get_width(), f.get_height()) * b.scale;
    return all(isfinite(v)) ? v : 0;
}

kernel void newLayered(device const float4* src  [[buffer(0)]],
                       device float4* dst        [[buffer(1)]],
                       constant NFrame& sf       [[buffer(2)]],
                       constant NFrame& df       [[buffer(3)]],
                       constant NBlur& b         [[buffer(4)]],
                       texture2d<float> vFG      [[texture(0)]],
                       texture2d<float> vBG      [[texture(1)]],
                       texture2d<float> matte    [[texture(2)]],
                       texture2d<float> plate    [[texture(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= df.width || gid.y >= df.height) return;
    if (matte.read(gid).x > 0.5) return;  // FG pixels: left to newReconstruct
    float4 cX = src[gid.y * sf.pitch + gid.x];
    float2 X = float2(gid) + 0.5;
    float4 layer[2];
    for (int fg = 0; fg < 2; fg++) {
        float2 v = fieldVel(fg ? vFG : vBG, b, X);
        int n = length(v) > 0.05 ? clamp(int(length(v) * b.ownDensity + 1.0), 1, 256) : 0;
        float2 d = n ? v / float(n) : 0;
        float4 sum = 0;
        float cnt = 0;
        for (int l = -n; l <= n; l++) {
            float2 Y = X + d * float(l);
            if (Y.x < 0 || Y.y < 0 || Y.x >= b.w || Y.y >= b.h) continue;
            sum += fg ? sampleImg(src, sf, Y) * matte.sample(pixLinear, Y).x : plate.sample(pixLinear, Y);
            cnt += 1;
        }
        layer[fg] = sum / max(cnt, 1.0f);
    }
    float4 blur = layer[1] + layer[0] * (1 - layer[1].w);
    dst[gid.y * df.pitch + gid.x] = b.mix == 1.0 ? blur : cX + (blur - cX) * b.mix;
}

// ---- Matte velocity cleaning (tracking resolution) ---------------------------------------------------------------

// Per-frame velocity at t from the two displacement fields (same modes as newVelocity), in tracking pixels.
kernel void newVelocityLow(texture2d<float> dPrev [[texture(0)]], texture2d<float> dNext [[texture(1)]],
                           texture2d<float, access::write> v [[texture(2)]], constant int& mode [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= v.get_width() || gid.y >= v.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(v.get_width(), v.get_height());
    float2 p = dPrev.sample(normLinear, uv).xy, n = dNext.sample(normLinear, uv).xy;
    v.write(float4(mode == 0 ? 0.5 * (p - n) : mode == 1 ? -n : p, 0, 0), gid);
}

// Push-pull seeds for one layer: velocity where the eroded (blurred) matte is clearly inside that layer.
kernel void newVelocitySeed(texture2d<float> v [[texture(0)]], texture2d<float> mBlur [[texture(1)]],
                            texture2d<float, access::write> B [[texture(2)]], texture2d<float, access::write> C [[texture(3)]],
                            constant int& foreground [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= v.get_width() || gid.y >= v.get_height()) return;
    float m = mBlur.read(gid).x;
    bool seed = foreground ? m > 0.99f : m < 0.01f;
    B.write(seed ? float4(v.read(gid).xy, 0, 0) : float4(0), gid);
    C.write(seed ? 1.0f : 0.0f, gid);
}

// Cleaned velocity: the foreground field inside the matte, the background field outside.
kernel void newVelocityPick(texture2d<float> vFG [[texture(0)]], texture2d<float> vBG [[texture(1)]],
                            texture2d<float> matte [[texture(2)]], texture2d<float, access::write> v [[texture(3)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= v.get_width() || gid.y >= v.get_height()) return;
    v.write(float4((matte.read(gid).x > 0.5f ? vFG : vBG).read(gid).xy, 0, 0), gid);
}

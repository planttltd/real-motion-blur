// Real Motion Blur: automatic optical-flow motion blur for After Effects, Apple silicon / Metal only.
#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"
#include "AE_EffectSuites.h"
#include "adobesdk/DrawbotSuite.h"
#import "RMBCore.h"
#include "LogoPath.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

#define NAME "Real Motion Blur"
#define MATCH_NAME "RPLT RealMotionBlur"
#define VEC_NAME "Real Motion Blur Vectors"
#define VEC_MATCH_NAME "RPLT RealMotionBlurVectors"

static const int kNumPts = 12;  // track points
enum {
    P_INPUT = 0, P_LOGO, P_LEGACY, P_AMOUNT, P_MOTION_SRC, P_SENSITIVITY, P_TRACK_FRAME, P_TRACK_ENHANCE, P_TRACK_RGBA,
    P_IMAGE_TRACK, P_FIRST_MASK, P_LAST_MASK, P_DISPLAY,
    P_FG_START, P_FG_MATTE, P_FG_MATTE_CHAN, P_FG_SHRINK, P_FG_AMOUNT, P_FG_SENSITIVITY,
    P_FG_IMAGE_TRACK, P_FG_FIRST_MASK, P_FG_LAST_MASK, P_FG_END,
    P_PTS_START, P_PT0, P_PTS_END = P_PT0 + 2 * kNumPts, P_NUM  // point i: P_PT0 + 2i "Use", + 1 "Pos"
};
enum {
    ID_AMOUNT = 1, ID_MOTION_SRC, ID_SENSITIVITY, ID_LOGO, ID_LEGACY, ID_TRACK_FRAME, ID_TRACK_RGBA, ID_DISPLAY,
    ID_FG_START, ID_FG_MATTE, ID_FG_MATTE_CHAN, ID_FG_SHRINK, ID_FG_AMOUNT, ID_FG_SENSITIVITY, ID_FG_END,
    ID_TRACK_ENHANCE, ID_IMAGE_TRACK, ID_FIRST_MASK, ID_LAST_MASK, ID_FG_IMAGE_TRACK, ID_FG_FIRST_MASK,
    ID_FG_LAST_MASK, ID_PTS_START, ID_PTS_END, ID_PT0 = 100  // point i: ID_PT0 + 2i, + 1
};
// Vectors effect.
enum { V_INPUT = 0, V_LOGO, V_VECTORS, V_LINEAR, V_AMOUNT, V_MAX_DISPLACE, V_SCALE_X, V_SCALE_Y, V_NUM };
enum { VID_LOGO = 1, VID_VECTORS, VID_AMOUNT, VID_MAX_DISPLACE, VID_SCALE_X, VID_SCALE_Y, VID_LINEAR };
static const A_short kLogoPad = 8;

// AE's GPU frames (SMART_RENDER_GPU) are off: in AE 26.5, any plugin render that checks out GPU frames, even one that
// does nothing else, makes the export fail ("Error Code: 3") when Shift Channels is the next effect on the layer
// (reproduced in a bare comp: solid -> this -> Shift Channels; Adobe's own GPU effects are fine there). The CPU entry
// point still renders everything on Metal; only the frame hand-off goes through memory. To re-test, flip this and
// the GPU_RENDER_F32 bit in the PiPL's AE_Effect_Global_OutFlags_2.
static const bool kAEGPUFrames = false;

#define LogError RMBLog

// Checkout ids (index + 1), each at t-1, t, t+1: [0..2] input, [3..5] motion source, [6..8] FG matte.
static const int kNumCheckouts = 9;

struct PreRenderData {
    RMBSettings settings;
    bool have[kNumCheckouts];
};
struct VecPreRenderData {
    RMBVectorSettings settings;
    bool have[2];  // checkout 1: input, 2: vectors (current time only)
};

static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    out_data->my_version = PF_VERSION(1, 0, 0, PF_Stage_DEVELOP, 1);
    // Must match AE_Effect_Global_OutFlags(_2) in RealMotionBlurPiPL.r.
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_CUSTOM_UI;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                           PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT | PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                           (kAEGPUFrames ? PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 : 0);
    return PF_Err_NONE;
}

static PF_Err VecGlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    out_data->my_version = PF_VERSION(1, 0, 0, PF_Stage_DEVELOP, 1);
    // Must match the Vectors PiPL in RealMotionBlurPiPL.r.
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_CUSTOM_UI;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                           PF_OutFlag2_SUPPORTS_THREADED_RENDERING | (kAEGPUFrames ? PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 : 0);
    return PF_Err_NONE;
}

static PF_Err AddPath(PF_InData* in_data, const char* name, A_long id)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_PATH;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.path_d.dephault = 0;  // none
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

// Logo header: a data-less param whose Effect Controls area we draw ourselves (DrawLogo).
static PF_Err AddLogo(PF_InData* in_data, A_long id)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_NO_DATA;
    def.ui_flags = PF_PUI_CONTROL;
    def.ui_width = (A_short)kLogoW + 2 * kLogoPad;
    def.ui_height = (A_short)kLogoH + 2 * kLogoPad;
    def.uu.id = id;
    ERR(PF_ADD_PARAM(in_data, -1, &def));
    PF_CustomUIInfo ci;
    AEFX_CLR_STRUCT(ci);
    ci.events = PF_CustomEFlag_EFFECT;
    ERR((*in_data->inter.register_ui)(in_data->effect_ref, &ci));
    return err;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;
    ERR(AddLogo(in_data, ID_LOGO));

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Legacy", FALSE, 0, ID_LEGACY);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blur Amount", -2, 10, 0, 1, 0.5, PF_Precision_HUNDREDTHS, 0, 0, ID_AMOUNT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Motion Src", PF_LayerDefault_NONE, ID_MOTION_SRC);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Motion Sensitivity", 0, 100, 0, 100, 70, PF_Precision_TENTHS, 0, 0, ID_SENSITIVITY);
    // Layer controls: a foreground matte splits the frame into FG1 and background layers.
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Track Frame", 3, 1, "Auto|Previous|Next", ID_TRACK_FRAME);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Track Enhance", 2, 1, "None|Delinearize", ID_TRACK_ENHANCE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Track RGB+A", FALSE, 0, ID_TRACK_RGBA);
    // Guides for the main (background) layer: masks on this layer and track points pin or drive its motion.
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Image Track", TRUE, 0, ID_IMAGE_TRACK);
    ERR(AddPath(in_data, "First Mask", ID_FIRST_MASK));
    ERR(AddPath(in_data, "Last Mask", ID_LAST_MASK));
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Display", 4, 1, "Color Src Blurred|Color Src UnBlurred|FG 1 Blurred|BG Blurred", ID_DISPLAY);
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("FG1 Settings", ID_FG_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("FG1 Matte", PF_LayerDefault_NONE, ID_FG_MATTE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("FG1 Matte Chan", 4, 1, "Red|Green|Blue|Alpha", ID_FG_MATTE_CHAN);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("FG1 Inv Matte Shrink", 0, 10, 0, 10, 0, PF_Precision_TENTHS, 0, 0, ID_FG_SHRINK);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("FG1 Blur Amt", -2, 10, 0, 1, 0.5, PF_Precision_HUNDREDTHS, 0, 0, ID_FG_AMOUNT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("FG1 Sensitivity", 0, 100, 0, 100, 70, PF_Precision_TENTHS, 0, 0, ID_FG_SENSITIVITY);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("FG1 Image Track", TRUE, 0, ID_FG_IMAGE_TRACK);
    ERR(AddPath(in_data, "FG1 First Mask", ID_FG_FIRST_MASK));
    ERR(AddPath(in_data, "FG1 Last Mask", ID_FG_LAST_MASK));
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_FG_END);
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Track Points", ID_PTS_START);
    for (int i = 0; i < kNumPts && !err; i++) {
        char name[32];
        snprintf(name, sizeof name, "Pt %d: Use", i + 1);
        AEFX_CLR_STRUCT(def);
        PF_ADD_POPUP(name, 3, 1, "Don't Use|FG1|BG", ID_PT0 + 2 * i);
        snprintf(name, sizeof name, "Pt %d: Pos", i + 1);
        AEFX_CLR_STRUCT(def);
        PF_ADD_POINT(name, 50, 50, FALSE, ID_PT0 + 2 * i + 1);
    }
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_PTS_END);
    out_data->num_params = P_NUM;
    return err;
}

static PF_Err VecParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;
    ERR(AddLogo(in_data, VID_LOGO));
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Motion Vectors", PF_LayerDefault_NONE, VID_VECTORS);
    // For projects that linearise the working space: AE then hands over the vector layer linearised too.
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Vectors are linear", FALSE, 0, VID_LINEAR);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blur Amt", -2, 10, 0, 1, 0.5, PF_Precision_HUNDREDTHS, 0, 0, VID_AMOUNT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Max Displace", 0, 10000, 0, 256, 64, PF_Precision_TENTHS, 0, 0, VID_MAX_DISPLACE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Vec Scale X", -2, 10, -2, 10, 1, PF_Precision_HUNDREDTHS, 0, 0, VID_SCALE_X);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Vec Scale Y", -2, 10, -2, 10, 1, PF_Precision_HUNDREDTHS, 0, 0, VID_SCALE_Y);
    out_data->num_params = V_NUM;
    return err;
}

// Value of a non-layer param at time t (default: now). The copy stays valid after check-in.
static PF_ParamDef ReadParam(PF_InData* in_data, A_long index, A_long t = -1)
{
    PF_ParamDef p;
    AEFX_CLR_STRUCT(p);
    if (!PF_CHECKOUT_PARAM(in_data, index, t < 0 ? in_data->current_time : t, in_data->time_step, in_data->time_scale, &p))
        PF_CHECKIN_PARAM(in_data, &p);
    return p;
}

static bool Empty(const PF_LRect& r) { return r.right <= r.left || r.bottom <= r.top; }

// ---- Guides: track points and mask samples, paired between now and a neighbour frame ----

struct Pt { double x, y; };

// m points at equal arc length along a polyline.
static std::vector<Pt> Resample(const std::vector<Pt>& p, int m)
{
    std::vector<double> cum = { 0 };
    for (size_t i = 1; i < p.size(); i++) cum.push_back(cum.back() + std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y));
    if (p.empty() || cum.back() < 0.001) return std::vector<Pt>(m, p.empty() ? Pt{ 0, 0 } : p[0]);
    std::vector<Pt> out;
    size_t j = 0;
    for (int k = 0; k < m; k++) {
        double s = cum.back() * k / (m - 1);
        while (j + 2 < cum.size() && cum[j + 1] < s) j++;
        double f = cum[j + 1] > cum[j] ? std::clamp((s - cum[j]) / (cum[j + 1] - cum[j]), 0.0, 1.0) : 0;
        out.push_back({ p[j].x + (p[j + 1].x - p[j].x) * f, p[j].y + (p[j + 1].y - p[j].y) * f });
    }
    return out;
}

struct PathWalker {
    PF_InData* in_data;
    PF_PathDataSuite1* pd;
    PF_PathOutlinePtr path;
    double length(A_long seg)
    {
        PF_PathSegPrepPtr prep = nullptr;
        PF_FpLong len = 0;
        pd->PF_PathPrepareSegLength(in_data->effect_ref, path, seg, 512, &prep);
        pd->PF_PathGetSegLength(in_data->effect_ref, path, seg, &prep, &len);
        pd->PF_PathCleanupSegLength(in_data->effect_ref, path, seg, &prep);
        return len;
    }
    // m points at equal arc length along segment seg (m >= 2).
    std::vector<Pt> walk(A_long seg, int m)
    {
        PF_PathSegPrepPtr prep = nullptr;
        PF_FpLong len = 0;
        pd->PF_PathPrepareSegLength(in_data->effect_ref, path, seg, 512, &prep);
        pd->PF_PathGetSegLength(in_data->effect_ref, path, seg, &prep, &len);
        std::vector<Pt> out(m);
        for (int k = 0; k < m; k++) pd->PF_PathEvalSegLength(in_data->effect_ref, path, &prep, seg, len * k / (m - 1), &out[k].x, &out[k].y);
        pd->PF_PathCleanupSegLength(in_data->effect_ref, path, seg, &prep);
        return out;
    }
};

// One mask at now (P) and at the neighbour (Q), sampled about every pixel of arc length. Segments are paired by
// index; when the vertex counts differ, the unpaired tails are each resampled to the same number of points.
static void MaskPairs(PathWalker P, PathWalker Q, std::vector<RMBTrackPair>& out)
{
    A_long nP = 0, nQ = 0;
    P.pd->PF_PathNumSegments(P.in_data->effect_ref, P.path, &nP);
    Q.pd->PF_PathNumSegments(Q.in_data->effect_ref, Q.path, &nQ);
    A_long c = nP == nQ ? nP : std::max<A_long>(0, std::min(nP, nQ) - 1);
    auto count = [](double len) { int m = int(std::floor(len)) + 1; return m < 3 ? 2 : std::min(m, 100000); };
    auto pair = [&](const std::vector<Pt>& a, const std::vector<Pt>& b) {
        for (size_t k = 0; k < a.size(); k++) out.push_back({ float(a[k].x), float(a[k].y), float(b[k].x), float(b[k].y) });
    };
    for (A_long i = 0; i < c; i++) {
        int m = count(P.length(i));
        pair(P.walk(i, m), Q.walk(i, m));
    }
    if (nP == nQ) return;
    auto tail = [&](PathWalker& w, A_long n) {
        std::vector<Pt> d;
        for (A_long i = c; i < n; i++) { auto s = w.walk(i, std::max(2, int(std::floor(w.length(i))) + 1)); d.insert(d.end(), s.begin(), s.end()); }
        if (d.empty()) {  // no segments left: the last vertex
            PF_PathVertex v;
            w.pd->PF_PathVertexInfo(w.in_data->effect_ref, w.path, c, &v);
            d.push_back({ v.x, v.y });
        }
        return d;
    };
    std::vector<Pt> dP = tail(P, nP), dQ = tail(Q, nQ);
    int m = std::max(2, int(dP.size()));
    pair(Resample(dP, m), Resample(dQ, m));
}

// Masks first..last (by their order on the layer; Last unset = First only), each paired between now and tn.
// Note: masks are used whatever their opacity (a mask below 100% opacity could be skipped, which needs AEGP access)
static void MasksPairs(PF_InData* in_data, PF_OutData* out_data, A_long firstP, A_long lastP, A_long tn,
                       std::vector<RMBTrackPair>& out)
{
    PF_PathID first = ReadParam(in_data, firstP).u.path_d.path_id, last = ReadParam(in_data, lastP).u.path_d.path_id;
    if (first == PF_PathID_NONE) return;
    AEFX_SuiteScoper<PF_PathQuerySuite1> pq(in_data, kPFPathQuerySuite, kPFPathQuerySuiteVersion1, out_data);
    AEFX_SuiteScoper<PF_PathDataSuite1> pd(in_data, kPFPathDataSuite, kPFPathDataSuiteVersion1, out_data);
    A_long num = 0, lo = -1, hi = -1;
    pq->PF_NumPaths(in_data->effect_ref, &num);
    for (A_long i = 0; i < num; i++) {
        PF_PathID id = PF_PathID_NONE;
        pq->PF_PathInfo(in_data->effect_ref, i, &id);
        if (id == first) lo = i;
        if (id == last) hi = i;
    }
    if (lo < 0) return;
    if (hi < 0) hi = lo;
    if (hi < lo) std::swap(lo, hi);
    for (A_long i = lo; i <= hi; i++) {
        PF_PathID id = PF_PathID_NONE;
        PF_PathOutlinePtr p = nullptr, q = nullptr;
        pq->PF_PathInfo(in_data->effect_ref, i, &id);
        pq->PF_CheckoutPath(in_data->effect_ref, id, in_data->current_time, in_data->time_step, in_data->time_scale, &p);
        pq->PF_CheckoutPath(in_data->effect_ref, id, tn, in_data->time_step, in_data->time_scale, &q);
        if (p && q) MaskPairs({ in_data, (PF_PathDataSuite1*)pd.get(), p }, { in_data, (PF_PathDataSuite1*)pd.get(), q }, out);
        if (p) pq->PF_CheckinPath(in_data->effect_ref, id, FALSE, p);
        if (q) pq->PF_CheckinPath(in_data->effect_ref, id, FALSE, q);
    }
}

// Points and masks for both layers, against the previous and the next frame. Positions are in the layer's pixels
// at the current resolution (AE scales point values for downsampling).
static void ReadGuides(PF_InData* in_data, PF_OutData* out_data, RMBSettings& s)
{
    s.guides[RMBLayerBG].imageTrack = ReadParam(in_data, P_IMAGE_TRACK).u.bd.value != 0;
    s.guides[RMBLayerFG].imageTrack = ReadParam(in_data, P_FG_IMAGE_TRACK).u.bd.value != 0;
    for (int dir = -1; dir <= 1; dir += 2) {
        A_long tn = in_data->current_time + dir * in_data->time_step;
        auto list = [&](int layer) -> std::vector<RMBTrackPair>& { return dir < 0 ? s.guides[layer].prev : s.guides[layer].next; };
        for (int i = 0; i < kNumPts; i++) {
            A_long use = ReadParam(in_data, P_PT0 + 2 * i).u.pd.value;  // 1 Don't Use, 2 FG1, 3 BG
            if (use < 2) continue;
            PF_ParamDef a = ReadParam(in_data, P_PT0 + 2 * i + 1), b = ReadParam(in_data, P_PT0 + 2 * i + 1, tn);
            list(use == 2 ? RMBLayerFG : RMBLayerBG).push_back({ float(FIX_2_FLOAT(a.u.td.x_value)), float(FIX_2_FLOAT(a.u.td.y_value)),
                                                                 float(FIX_2_FLOAT(b.u.td.x_value)), float(FIX_2_FLOAT(b.u.td.y_value)) });
        }
        MasksPairs(in_data, out_data, P_FIRST_MASK, P_LAST_MASK, tn, list(RMBLayerBG));
        MasksPairs(in_data, out_data, P_FG_FIRST_MASK, P_FG_LAST_MASK, tn, list(RMBLayerFG));
    }
}

// Tracking needs whole frames, so always ask for the full layer, not just the requested rect.
static PF_RenderRequest FullRequest(PF_InData* in_data, PF_PreRenderExtra* extra)
{
    PF_RenderRequest req = extra->input->output_request;
    req.rect.left = req.rect.top = 0;
    req.rect.right = in_data->width * in_data->downsample_x.num / in_data->downsample_x.den;
    req.rect.bottom = in_data->height * in_data->downsample_y.num / in_data->downsample_y.den;
    req.channel_mask = PF_ChannelMask_ARGB;
    return req;
}

static PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    auto* pd = new PreRenderData();
    pd->settings.amount = (float)ReadParam(in_data, P_AMOUNT).u.fs_d.value;
    pd->settings.sensitivity = (float)ReadParam(in_data, P_SENSITIVITY).u.fs_d.value / 100.0f;
    pd->settings.legacy = ReadParam(in_data, P_LEGACY).u.bd.value != 0;
    pd->settings.hq = in_data->quality == PF_Quality_HI;  // Legacy: Best tracks at 1/2 res + bilinear, Draft 1/4 + nearest
    pd->settings.trackFrame = ReadParam(in_data, P_TRACK_FRAME).u.pd.value - 1;  // popups are 1-based
    pd->settings.trackRGBA = ReadParam(in_data, P_TRACK_RGBA).u.bd.value != 0;
    pd->settings.trackDelinearize = ReadParam(in_data, P_TRACK_ENHANCE).u.pd.value == 2;
    pd->settings.display = ReadParam(in_data, P_DISPLAY).u.pd.value - 1;
    pd->settings.matteChannel = ReadParam(in_data, P_FG_MATTE_CHAN).u.pd.value - 1;
    pd->settings.matteShrink = (float)ReadParam(in_data, P_FG_SHRINK).u.fs_d.value *
                               in_data->downsample_x.num / (float)in_data->downsample_x.den;
    pd->settings.fgAmount = (float)ReadParam(in_data, P_FG_AMOUNT).u.fs_d.value;
    pd->settings.fgSensitivity = (float)ReadParam(in_data, P_FG_SENSITIVITY).u.fs_d.value / 100.0f;
    ReadGuides(in_data, out_data, pd->settings);
    extra->output->pre_render_data = pd;
    extra->output->delete_pre_render_data_func = [](void* p) { delete (PreRenderData*)p; };

    PF_RenderRequest req = FullRequest(in_data, extra);

    for (int i = 0; i < kNumCheckouts && !err; i++) {
        A_long layer = i < 3 ? P_INPUT : i < 6 ? P_MOTION_SRC : P_FG_MATTE;
        A_long t = in_data->current_time + (i % 3 - 1) * in_data->time_step;
        PF_CheckoutResult r;
        AEFX_CLR_STRUCT(r);
        err = extra->cb->checkout_layer(in_data->effect_ref, layer, i + 1, &req, t, in_data->time_step,
                                        in_data->time_scale, &r);
        if (err)
            LogError("pre-render: checkout_layer %d (param %d, time %d/%u) failed: %d; request %d,%d %dx%d", i + 1,
                     (int)layer, (int)t, (unsigned)in_data->time_scale, (int)err, (int)req.rect.left, (int)req.rect.top,
                     (int)(req.rect.right - req.rect.left), (int)(req.rect.bottom - req.rect.top));
        pd->have[i] = !err && !Empty(r.result_rect);
        if (i == 1) {
            extra->output->result_rect = r.result_rect;
            extra->output->max_result_rect = r.max_result_rect;
        }
    }
    extra->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS | (kAEGPUFrames ? PF_RenderOutputFlag_GPU_RENDER_POSSIBLE : 0);
    return err;
}

static PF_Err VecPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    auto* pd = new VecPreRenderData();
    float k = in_data->downsample_x.num / (float)in_data->downsample_x.den;  // vectors are full-resolution pixels
    pd->settings.amount = (float)ReadParam(in_data, V_AMOUNT).u.fs_d.value;
    pd->settings.maxDisplace = (float)ReadParam(in_data, V_MAX_DISPLACE).u.fs_d.value;
    pd->settings.scaleX = (float)ReadParam(in_data, V_SCALE_X).u.fs_d.value * k;
    pd->settings.scaleY = (float)ReadParam(in_data, V_SCALE_Y).u.fs_d.value * k;
    pd->settings.hq = in_data->quality == PF_Quality_HI;
    pd->settings.linear = ReadParam(in_data, V_LINEAR).u.bd.value != 0;
    extra->output->pre_render_data = pd;
    extra->output->delete_pre_render_data_func = [](void* p) { delete (VecPreRenderData*)p; };
    PF_RenderRequest req = FullRequest(in_data, extra);  // the hole fill needs the whole vector frame
    for (int i = 0; i < 2 && !err; i++) {
        PF_CheckoutResult r;
        AEFX_CLR_STRUCT(r);
        err = extra->cb->checkout_layer(in_data->effect_ref, i ? V_VECTORS : V_INPUT, i + 1, &req, in_data->current_time,
                                        in_data->time_step, in_data->time_scale, &r);
        if (err) LogError("vectors pre-render: checkout_layer %d failed: %d", i + 1, (int)err);
        pd->have[i] = !err && !Empty(r.result_rect);
        if (i == 0) {
            extra->output->result_rect = r.result_rect;
            extra->output->max_result_rect = r.max_result_rect;
        }
    }
    extra->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS | (kAEGPUFrames ? PF_RenderOutputFlag_GPU_RENDER_POSSIBLE : 0);
    return err;
}

// ---- CPU path: AE hands us ARGB 8/16/32 bpc worlds; run the same Metal core on copies. ----


// Rows in parallel bands (these copies are most of the CPU path's own cost).
template <typename F> static void ForRows(A_long h, F&& rowFn)
{
    const size_t bands = 16;
    dispatch_apply(bands, DISPATCH_APPLY_AUTO, ^(size_t b) {
        for (A_long y = A_long(b * h / bands); y < A_long((b + 1) * h / bands); y++) rowFn(y);
    });
}

template <typename Pix, typename Ch>
static void CopyIn(const PF_EffectWorld* w, float* dst, float max)
{
    ForRows(w->height, [&](A_long y) {
        const Pix* row = (const Pix*)((const char*)w->data + y * w->rowbytes);
        float* d = dst + (size_t)y * w->width * 4;
        for (A_long x = 0; x < w->width; x++, d += 4) {
            d[0] = row[x].blue / max; d[1] = row[x].green / max;
            d[2] = row[x].red / max;  d[3] = row[x].alpha / max;
        }
    });
}

template <typename Pix, typename Ch>
static void CopyOut(const float* src, PF_EffectWorld* w, float max)
{
    auto q = [max](float v) -> Ch {
        if (max == 1.0f) return (Ch)v;
        v = v * max + 0.5f;
        return (Ch)(v < 0 ? 0 : v > max ? max : v);
    };
    ForRows(w->height, [&](A_long y) {
        Pix* row = (Pix*)((char*)w->data + y * w->rowbytes);
        const float* s = src + (size_t)y * w->width * 4;
        for (A_long x = 0; x < w->width; x++, s += 4) {
            row[x].blue = q(s[0]); row[x].green = q(s[1]);
            row[x].red = q(s[2]);  row[x].alpha = q(s[3]);
        }
    });
}

static RMBFrame Upload(id<MTLDevice> dev, PF_EffectWorld* w, PF_PixelFormat fmt)
{
    if (!w) return { nil, 0, 0, 0 };
    RMBFrame f = { [dev newBufferWithLength:(size_t)w->width * w->height * 16 options:MTLResourceStorageModeShared],
                   (uint32_t)w->width, (uint32_t)w->width, (uint32_t)w->height };
    float* p = (float*)f.buf.contents;
    if (fmt == PF_PixelFormat_ARGB128) CopyIn<PF_PixelFloat, PF_FpShort>(w, p, 1.0f);
    else if (fmt == PF_PixelFormat_ARGB64) CopyIn<PF_Pixel16, A_u_short>(w, p, PF_MAX_CHAN16);
    else CopyIn<PF_Pixel8, A_u_char>(w, p, PF_MAX_CHAN8);
    return f;
}

static RMBFrame Wrap(PF_InData* in_data, const PF_GPUDeviceSuite1* gpu, PF_EffectWorld* w)
{
    void* mem = nullptr;
    if (!w || gpu->GetGPUWorldData(in_data->effect_ref, w, &mem) || !mem) return { nil, 0, 0, 0 };
    return { (__bridge id<MTLBuffer>)mem, (uint32_t)(w->rowbytes / 16), (uint32_t)w->width, (uint32_t)w->height };
}

// Checks out layer checkouts 1..n (where have[i]), then runs render(device, queue, frames, dst, wait) on Metal:
// directly on AE's GPU frames, or on float copies of the CPU worlds. frames[i] is nil where not checked out.
template <typename F>
static PF_Err SmartRenderWith(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool isGPU,
                              int n, const bool* have, int required, F&& render)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_EffectWorld* worlds[kNumCheckouts] = {};
    for (int i = 0; i < n && !err; i++)
        if (have[i] && (err = extra->cb->checkout_layer_pixels(in_data->effect_ref, i + 1, &worlds[i])))
            LogError("render: checkout_layer_pixels %d failed: %d", i + 1, (int)err);
    PF_EffectWorld* output = nullptr;
    if (!err && (err = extra->cb->checkout_output(in_data->effect_ref, &output))) LogError("render: checkout_output failed: %d", (int)err);
    if (!err && worlds[required] && output) {  // sizes, for the log when something below fails
        for (int i = 0; i < n; i++)
            if (worlds[i] && (worlds[i]->width != output->width || worlds[i]->height != output->height))
                LogError("render: checkout %d is %dx%d, output %dx%d", i + 1, (int)worlds[i]->width,
                         (int)worlds[i]->height, (int)output->width, (int)output->height);
    }

    if (!err && worlds[required] && output) {
        if (isGPU) {
            AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
            PF_GPUDeviceInfo info;
            AEFX_CLR_STRUCT(info);
            ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
            if (err) LogError("render (GPU): GetDeviceInfo failed: %d", (int)err);
            RMBFrame f[kNumCheckouts] = {};
            for (int i = 0; i < n; i++) f[i] = Wrap(in_data, gpu.get(), worlds[i]);
            // Waits for the GPU before returning, so AE never sees a frame still in flight.
            id<MTLDevice> dev = (__bridge id<MTLDevice>)info.devicePV;
            id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)info.command_queuePV;
            if (!err && !render(dev, q, f, Wrap(in_data, gpu.get(), output), true)) {
                LogError("render (GPU): core render failed, output %dx%d", (int)output->width, (int)output->height);
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        } else {
            static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            static id<MTLCommandQueue> queue = [dev newCommandQueue];
            AEFX_SuiteScoper<PF_WorldSuite2> ws(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
            PF_PixelFormat fmt = PF_PixelFormat_ARGB32;
            ERR(ws->PF_GetPixelFormat(output, &fmt));
            @autoreleasepool {
                RMBFrame f[kNumCheckouts] = {};
                for (int i = 0; i < n; i++) f[i] = Upload(dev, worlds[i], fmt);
                RMBFrame dst = { [dev newBufferWithLength:(size_t)output->width * output->height * 16
                                                  options:MTLResourceStorageModeShared],
                                 (uint32_t)output->width, (uint32_t)output->width, (uint32_t)output->height };
                if (!err && render(dev, queue, f, dst, true)) {
                    const float* p = (const float*)dst.buf.contents;
                    if (fmt == PF_PixelFormat_ARGB128) CopyOut<PF_PixelFloat, PF_FpShort>(p, output, 1.0f);
                    else if (fmt == PF_PixelFormat_ARGB64) CopyOut<PF_Pixel16, A_u_short>(p, output, PF_MAX_CHAN16);
                    else CopyOut<PF_Pixel8, A_u_char>(p, output, PF_MAX_CHAN8);
                } else if (!err) {
                    LogError("render (CPU): core render failed, output %dx%d", (int)output->width, (int)output->height);
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                }
            }
        }
    }
    for (int i = 0; i < n; i++)
        if (have[i]) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, i + 1));
    return err;
}

static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool isGPU)
{
    auto* pd = (PreRenderData*)extra->input->pre_render_data;
    if (!pd) return PF_Err_INTERNAL_STRUCT_DAMAGED;
    return SmartRenderWith(in_data, out_data, extra, isGPU, kNumCheckouts, pd->have, 1,
        [pd](id<MTLDevice> dev, id<MTLCommandQueue> q, const RMBFrame* f, const RMBFrame& dst, bool wait) {
            RMBInputs in = {};
            for (int i = 0; i < 3; i++) {
                in.src[i] = f[i];
                in.track[i] = f[3 + i];  // nil without Motion Src: the core tracks src
                in.matte[i] = f[6 + i];
            }
            return RMBRender(dev, q, in, dst, pd->settings, wait);
        });
}

static PF_Err VecSmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool isGPU)
{
    auto* pd = (VecPreRenderData*)extra->input->pre_render_data;
    if (!pd) return PF_Err_INTERNAL_STRUCT_DAMAGED;
    return SmartRenderWith(in_data, out_data, extra, isGPU, 2, pd->have, 0,
        [pd](id<MTLDevice> dev, id<MTLCommandQueue> q, const RMBFrame* f, const RMBFrame& dst, bool wait) {
            return RMBRenderVectors(dev, q, f[0], f[1], dst, pd->settings, wait);
        });
}

// Draws assets/logo.svg (via LogoPath.h) in white at the top of the Effect Controls panel.
static PF_Err DrawLogo(PF_InData* in_data, PF_OutData* out_data, PF_EventExtra* ev)
{
    if (ev->e_type != PF_Event_DRAW || ev->effect_win.area != PF_EA_CONTROL) return PF_Err_NONE;
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    AEFX_SuiteScoper<PF_EffectCustomUISuite2> ui(in_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion2, out_data);
    AEFX_SuiteScoper<DRAWBOT_DrawbotSuiteCurrent> draw(in_data, kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_VersionCurrent, out_data);
    AEFX_SuiteScoper<DRAWBOT_SupplierSuiteCurrent> sup(in_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, out_data);
    AEFX_SuiteScoper<DRAWBOT_SurfaceSuiteCurrent> surf(in_data, kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_VersionCurrent, out_data);
    AEFX_SuiteScoper<DRAWBOT_PathSuiteCurrent> path(in_data, kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_VersionCurrent, out_data);

    DRAWBOT_DrawRef dref = nullptr;
    DRAWBOT_SupplierRef sref = nullptr;
    DRAWBOT_SurfaceRef surfref = nullptr;
    DRAWBOT_PathRef pref = nullptr;
    DRAWBOT_BrushRef bref = nullptr;
    ERR(ui->PF_GetDrawingReference(ev->contextH, &dref));
    ERR(draw->GetSupplier(dref, &sref));
    ERR(draw->GetSurface(dref, &surfref));
    ERR(sup->NewPath(sref, &pref));
    DRAWBOT_ColorRGBA white = { 1, 1, 1, 1 };
    ERR(sup->NewBrush(sref, &white, &bref));

    const float x0 = ev->effect_win.current_frame.left + kLogoPad, y0 = ev->effect_win.current_frame.top + kLogoPad;
    for (const LogoOp& o : kLogo) {
        if (err) break;
        switch (o.op) {
        case 'M': err = path->MoveTo(pref, x0 + o.p[0], y0 + o.p[1]); break;
        case 'L': err = path->LineTo(pref, x0 + o.p[0], y0 + o.p[1]); break;
        case 'C': {
            DRAWBOT_PointF32 c[3] = { { x0 + o.p[0], y0 + o.p[1] }, { x0 + o.p[2], y0 + o.p[3] }, { x0 + o.p[4], y0 + o.p[5] } };
            err = path->BezierTo(pref, &c[0], &c[1], &c[2]);
            break;
        }
        case 'Z': err = path->Close(pref); break;
        }
    }
    ERR(surf->FillPath(surfref, bref, pref, kDRAWBOT_FillType_EvenOdd));  // even-odd keeps the P/a counters open
    if (bref) ERR2(sup->ReleaseObject((DRAWBOT_ObjectRef)bref));
    if (pref) ERR2(sup->ReleaseObject((DRAWBOT_ObjectRef)pref));
    if (!err) ev->evt_out_flags = PF_EO_HANDLED_EVENT;
    return err;
}

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                    SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                                    const char* inHostVersion)
{
    PF_Err result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, NAME, MATCH_NAME, "Blur & Sharpen",
                                            AE_RESERVED_INFO, "EffectMain", "https://github.com/planttltd/real-motion-blur");
    if (!result)
        result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, VEC_NAME, VEC_MATCH_NAME, "Blur & Sharpen",
                                         AE_RESERVED_INFO, "VectorsMain", "https://github.com/planttltd/real-motion-blur");
    return result;
}

static PF_Err EffectMainImpl(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                             PF_LayerDef* output, void* extra);

// RMB_TRACE=1 in the environment logs every command and its result (diagnostics).
extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                       PF_LayerDef* output, void* extra)
{
    static const bool trace = getenv("RMB_TRACE") != nullptr;
    PF_Err err = EffectMainImpl(cmd, in_data, out_data, params, output, extra);
    if (trace) LogError("trace: command %d -> %d (time %d/%u)", (int)cmd, (int)err, in_data ? (int)in_data->current_time : -1,
                        in_data ? (unsigned)in_data->time_scale : 0);
    return err;
}

static PF_Err EffectMainImpl(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                             PF_LayerDef* output, void* extra)
{
    try {
        switch (cmd) {
        case PF_Cmd_ABOUT:
            PF_SPRINTF(out_data->return_msg, "%s v1.0\rOptical-flow motion blur (Metal).", NAME);
            return PF_Err_NONE;
        case PF_Cmd_GLOBAL_SETUP: return GlobalSetup(in_data, out_data);
        case PF_Cmd_PARAMS_SETUP: return ParamsSetup(in_data, out_data);
        case PF_Cmd_EVENT: return DrawLogo(in_data, out_data, (PF_EventExtra*)extra);
        case PF_Cmd_GPU_DEVICE_SETUP:
            // Only Metal; pipelines are built lazily per device in RMBCore.
            if (kAEGPUFrames && ((PF_GPUDeviceSetupExtra*)extra)->input->what_gpu == PF_GPU_Framework_METAL)
                out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
            return PF_Err_NONE;
        case PF_Cmd_GPU_DEVICE_SETDOWN: return PF_Err_NONE;
        case PF_Cmd_SMART_PRE_RENDER: return PreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
        case PF_Cmd_SMART_RENDER: return SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false);
        case PF_Cmd_SMART_RENDER_GPU: return SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true);
        default: return PF_Err_NONE;
        }
    } catch (const std::exception& e) {
        LogError("command %d: exception: %s", (int)cmd, e.what());
        return PF_Err_INTERNAL_STRUCT_DAMAGED;  // never throw into AE
    } catch (...) {
        LogError("command %d: unknown exception", (int)cmd);
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}

// Real Motion Blur Vectors: the same blur driven by a motion-vector layer instead of tracking.
extern "C" DllExport PF_Err VectorsMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                        PF_LayerDef* output, void* extra)
{
    try {
        switch (cmd) {
        case PF_Cmd_ABOUT:
            PF_SPRINTF(out_data->return_msg, "%s v1.0\rMotion blur from a motion-vector layer (Metal).", VEC_NAME);
            return PF_Err_NONE;
        case PF_Cmd_GLOBAL_SETUP: return VecGlobalSetup(in_data, out_data);
        case PF_Cmd_PARAMS_SETUP: return VecParamsSetup(in_data, out_data);
        case PF_Cmd_SMART_PRE_RENDER: return VecPreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
        case PF_Cmd_SMART_RENDER: return VecSmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false);
        case PF_Cmd_SMART_RENDER_GPU: return VecSmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true);
        default: return EffectMain(cmd, in_data, out_data, params, output, extra);  // logo, GPU setup
        }
    } catch (const std::exception& e) {
        LogError("vectors command %d: exception: %s", (int)cmd, e.what());
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    } catch (...) {
        LogError("vectors command %d: unknown exception", (int)cmd);
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}

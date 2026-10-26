# Real Motion Blur

Automatic optical-flow motion blur for Adobe After Effects. macOS on Apple silicon only: everything runs on the GPU
through Metal, and there is no Intel, Windows or CPU-only build. There is no ML anywhere: every stage is a classical
algorithm.

Real Motion Blur is open source and not affiliated with or endorsed by Adobe. Adobe and After Effects are trademarks of
Adobe Inc.

## Install

1. Download `RealMotionBlur-<version>.pkg` from the
   [Releases page](https://github.com/planttltd/real-motion-blur/releases).
2. Quit After Effects, double-click the package and follow the installer. It puts the effect into every copy of
   After Effects on the Mac. Run it again after installing a new After Effects version.
3. Start After Effects: the effect is under **Effect > Blur & Sharpen > Real Motion Blur**.

If macOS says the package "can't be opened" or "Apple could not verify" it, that release is not notarized. Open
**System Settings > Privacy & Security**, scroll to the message about RealMotionBlur, click **Open Anyway** and run
the installer again. A notarized release opens without this step.

To uninstall, open `/Library/Application Support/Real Motion Blur` in Finder and double-click
**Uninstall Real Motion Blur.command**.

Needs a Mac with Apple silicon, macOS 14 or later and After Effects 2026 (26.5 was used for development).

## Build from source

```bash
./build.sh test       # build and run the core self-check (no After Effects or SDK needed)
./build.sh            # build build/RealMotionBlur.plugin
./build.sh install    # build and copy into "/Applications/Adobe After Effects 2026/Plug-ins"
```

Requirements:
- Xcode with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).
- The After Effects SDK, which Adobe does not allow redistributing, so it is not in this repository. Download it from
  Adobe and set `AE_SDK` to its `Examples` folder:

```bash
AE_SDK=~/SDKs/AfterEffectsSDK/Examples ./build.sh install
```

Restart After Effects after installing: it only loads plugins at launch. A plugin built this way is ad-hoc signed,
which After Effects accepts for plugins built on the same Mac.

## Making the installer

`installer/build_pkg.sh` builds `build/RealMotionBlur-<version>.pkg` (version from `src/Info.plist`):

```bash
AE_SDK=~/SDKs/AfterEffectsSDK/Examples installer/build_pkg.sh
```

The package installs the plugin into `/Library/Application Support/Real Motion Blur` and its post-install script
copies it into every `/Applications/Adobe After Effects*/Plug-ins` folder. That is Adobe's recommended place for
effects that only run in After Effects, unlike the shared MediaCore folder, which Premiere Pro also loads.

For a release that opens with no Gatekeeper prompt, sign and notarize it. That needs an Apple Developer Program
membership with a **Developer ID Application** and a **Developer ID Installer** certificate:

```bash
xcrun notarytool store-credentials rmb-notary      # once: Apple ID, team ID and an app-specific password
AE_SDK=... SIGN_APP="Developer ID Application: <Team> (<ID>)" \
  SIGN_PKG="Developer ID Installer: <Team> (<ID>)" NOTARY_PROFILE=rmb-notary installer/build_pkg.sh
gh release create v1.0 build/RealMotionBlur-1.0.pkg --title "Real Motion Blur 1.0"
```

The script signs the plugin with the hardened runtime and a timestamp, signs the package, submits it to Apple's
notary service, waits, staples the ticket and checks the result with `spctl`.

## Modes

- **New** (default): better and faster than Legacy on every scene measured.
  - Motion is tracked against both neighbouring frames (Legacy's tracker, with its slow serial search replaced by a
    parallel one) and the two directions are averaged.
  - The blur gathers along each pixel's own motion. Where depth changes (a layer's alpha edge, or the FG1 matte),
    nearer content streaks over what is behind it.
  - With a matte, each side's motion is rebuilt from pixels well inside that side. Background pixels then show the
    foreground, blurred along its own motion, over the background with the foreground's hole filled in.
- **Legacy**: the classic look.
  - Pyramid Lucas–Kanade tracking with block-match propagation against one neighbour frame.
  - A centred gather blur along each pixel's own vector.
  - With an FG1 matte, the frame is split into foreground and background layers. Each is tracked on its own matted
    image, the background hidden behind the foreground is rebuilt from the neighbour frame and a push-pull fill, and
    the blurred layers are composited back to front.

## Controls

| Control | Meaning |
|---|---|
| Legacy | Classic algorithm instead of New (with two clean-ups: tracking reads the visible, premultiplied image, and each tracker step is capped at 3 px so flat areas beside strong edges cannot produce runaway vectors) |
| Blur Amount | Streak length as a fraction of one frame's motion (0.5 = 180° shutter, 1 = 360°). Negative values sharpen along the motion |
| Motion Src | Track motion on another layer instead (same size) |
| Motion Sensitivity | How large a motion the tracker can pick up. Lower it to reduce smearing where objects cross |
| Track Frame | Auto (skips a neighbour across a cut), Previous or Next |
| Track Enhance | Delinearize: tracks on the image raised to the power 2.2 (try it when tracking struggles) |
| Track RGB+A | Include alpha in the tracking image |
| Display | Blurred, unblurred, foreground layer only, or background layer only |
| FG1 Matte / Matte Chan | Layer and channel defining the foreground |
| FG1 Inv Matte Shrink | Grows the foreground matte (pixels) before the background is split off |
| FG1 Blur Amt / FG1 Sensitivity | Per-layer settings for the foreground (Legacy) |
| Image Track / FG1 Image Track | On: track the image, pinned to the layer's guides. Off: motion comes only from the guides (none without any). With it off, Track Frame Auto means Previous |
| First Mask / Last Mask (and FG1 ...) | A range of this layer's masks used as guides. Each mask is paired with itself on the neighbour frame along its arc length |
| Track Points | 12 points, each assigned to FG1 or BG. A point's motion between frames is a guide |

Guides are positions known on both frames. With Image Track on, the tracker takes the guided displacement exactly
at every pyramid level and spreads it through its smoothing. With Image Track off, the field is interpolated from
the guides by coarse-to-fine normalised convolution, and it fades to no motion away from them. In New mode, every
layer's guides apply to the one whole-frame track.

## Real Motion Blur Vectors

A second effect in the same bundle. It blurs along a motion-vector layer that you render elsewhere, instead of
tracking.
- Red is x and green is y; 0.5 means no motion and +y points up. A channel value of 0 or 1 stands for ±Max Displace
  pixels.
- Vectors count only where the vector layer is opaque. Elsewhere they are filled in from the surroundings by the
  same push-pull fill the layers use.
- Vec Scale X / Y scale each axis. Vectors are in full-resolution pixels and are rescaled for AE's resolution
  setting.
- Without a vector layer, the input passes through.

Vectors are data, not colour, so they must reach the effect unconverted:
- Interpret the vector footage with **Preserve RGB** (Interpret Footage > Color), or turn colour management off
  for it. Otherwise AE converts it to the working space and the vectors change.
- A working space with **Linearize Working Space** on also linearises the vector layer (0.5, no motion, arrives as
  about 0.21). Turn on **Vectors are linear** there: it re-encodes the values with the sRGB curve before decoding.
  It assumes an sRGB-curve working space. With another curve (for example gamma 2.4), use a gamma-encoded working
  space for this comp instead.
- 16 and 32 bpc vector renders keep the most precision. 8 bpc limits vectors to steps of Max Displace / 127.

AE gives effects straight (unpremultiplied) alpha. Both effects blur premultiplied and return straight alpha, so
transparent edges streak without dark fringes.

AE's Best quality tracks at ½ resolution with bilinear sampling. Draft quality tracks at ¼ resolution with nearest
sampling.

## Layout

- `src/RealMotionBlur.mm`: AE glue for both effects (parameters, track points and mask sampling, SmartFX checkout,
  GPU and CPU entry points, logo).
- `src/RMBCore.*`: AE-independent core: tracking, layers, blur, frame choice, GPU scratch pool.
- `src/RMBLegacy.metal`: the tracker and the gather blur.
- `src/RMBNew.metal`: New-mode velocity, matte cleanup and reconstruction blur.
- `src/RMBLayers.metal`: matte choke, layer split, warp, push-pull fill and compositing.
- `test/core_test.mm`: self-check, run by `./build.sh test`.
- `test/ae_e2e.jsx`: in-AE render test (both effects). Run it on an empty project; it reads back through `build/ae_e2e/e2e.aep`.
- `installer/`: the macOS installer package (`build_pkg.sh`, post-install script, uninstaller, installer pages).
- `assets/logo.svg` and `tools/svg_to_drawbot.py` generate `src/LogoPath.h`, which draws the Effect Controls header.

## Performance (M5 Max, per frame)

| | 1080p | 4K |
|---|---|---|
| Legacy | ~8.3 ms | ~22.5 ms |
| Legacy + matte | ~23 ms | ~61 ms |
| New | ~6.7 ms | ~21 ms |
| New + matte | ~9.6 ms | ~32 ms |

Quality against rendered ground truth (PSNR, 180° shutter; higher is better):

| Scene | Legacy | New |
|---|---|---|
| Disc over a panning photo | 49.50 dB | 50.92 dB |
| Same, with an FG matte (edges) | 46.96 dB | 47.14 dB |
| Screen layer over transparency, whip pan | 46.21 dB | 46.22 dB |
| Two pages closing at different rates | 41.98 dB | 43.68 dB |

Both effects take AE's frames through its CPU entry point and render them on Metal; the copies in and out run
in parallel. AE 26.5's own GPU frames are switched off: with them, any plugin render (even one that does nothing)
made exports fail with "Error Code: 3" when Shift Channels came next on the layer (see `kAEGPUFrames`).

Failures are logged to `~/Library/Logs/RealMotionBlur.log`. Setting `RMB_TRACE=1` in the environment (for example
for `aerender`) also logs every command AE sends and its result.

GPU scratch memory is pooled between frames and reused within a frame. The idle pool is capped at a quarter of the
GPU's recommended working set.

## Licence

MIT, see `LICENSE`. After Effects and its SDK are Adobe's and are not included.

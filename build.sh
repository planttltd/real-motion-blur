#!/bin/zsh
# ./build.sh            build build/RealMotionBlur.plugin
# ./build.sh test       build + run the core self-check
# ./build.sh install    build + copy into After Effects' Plug-ins folder
set -euo pipefail
cd "${0:A:h}"
# The After Effects SDK is not part of this repository: download it from Adobe and point AE_SDK at its Examples
# folder (only the plugin build needs it, not "test").
SDK=${AE_SDK:-}
AE_PLUGINS=${AE_PLUGINS:-"/Applications/Adobe After Effects 2026/Plug-ins"}
CXX=(clang++ -std=c++20 -O2 -fobjc-arc -arch arm64 -mmacosx-version-min=14.0 -Wall -Wno-unused-function)
FW=(-framework Foundation -framework Metal)
mkdir -p build

xcrun -sdk macosx metal -O3 -mmacosx-version-min=14.0 src/RMBLegacy.metal src/RMBNew.metal src/RMBLayers.metal -o build/RMBKernels.metallib

if [[ ${1:-} == test ]]; then
  "${CXX[@]}" test/core_test.mm src/RMBCore.mm "${FW[@]}" -framework ImageIO -framework CoreGraphics -o build/core_test
  ./build/core_test
  exit
fi

[[ -n $SDK ]] || { echo "set AE_SDK to the Examples folder of the After Effects SDK (see README)"; exit 1; }
B=build/RealMotionBlur.plugin/Contents
rm -rf build/RealMotionBlur.plugin
mkdir -p $B/MacOS $B/Resources
"${CXX[@]}" -bundle src/RealMotionBlur.mm src/RMBCore.mm \
  -I"$SDK/Headers" -I"$SDK/Headers/SP" -I"$SDK/Headers/adobesdk" -I"$SDK/Util" \
  "${FW[@]}" -o $B/MacOS/RealMotionBlur
xcrun Rez -d __MACH__ -useDF -i "$SDK/Headers" -i "$SDK/Resources" \
  -o $B/Resources/RealMotionBlur.rsrc src/RealMotionBlurPiPL.r
cp src/Info.plist $B/Info.plist
printf 'eFKTFXTC' > $B/PkgInfo
cp build/RMBKernels.metallib $B/Resources/
codesign --force --sign - build/RealMotionBlur.plugin
echo "built build/RealMotionBlur.plugin"

if [[ ${1:-} == install ]]; then
  rm -rf "$AE_PLUGINS/RealMotionBlur.plugin"
  cp -R build/RealMotionBlur.plugin "$AE_PLUGINS/"
  echo "installed to $AE_PLUGINS"
fi

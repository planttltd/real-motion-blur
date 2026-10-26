#!/bin/zsh
# Builds build/RealMotionBlur-<version>.pkg, the double-click installer for regular users.
#   AE_SDK=<After Effects SDK>/Examples installer/build_pkg.sh
# Optional, for a release that opens without Gatekeeper warnings (needs an Apple Developer ID):
#   SIGN_APP="Developer ID Application: <Team> (<ID>)"   signs the plugin (hardened runtime, timestamp)
#   SIGN_PKG="Developer ID Installer: <Team> (<ID>)"     signs the installer package
#   NOTARY_PROFILE=<name>   notarizes and staples (create it once: xcrun notarytool store-credentials <name>)
# Without them the plugin is ad-hoc signed and the package unsigned: it installs fine, but macOS asks the user to
# allow it once in System Settings > Privacy & Security.
set -euo pipefail
cd "${0:A:h}/.."
ID=com.plantt.realmotionblur
VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" src/Info.plist)
OUT=build/RealMotionBlur-$VERSION.pkg
STAGE=build/pkg/root
APPDIR="Library/Application Support/Real Motion Blur"

./build.sh
rm -rf build/pkg && mkdir -p "$STAGE/$APPDIR"
ditto build/RealMotionBlur.plugin "$STAGE/$APPDIR/RealMotionBlur.plugin"
cp installer/uninstall.sh "$STAGE/$APPDIR/Uninstall Real Motion Blur.command"
cp LICENSE "$STAGE/$APPDIR/LICENSE.txt"
if [[ -n ${SIGN_APP:-} ]]; then
  codesign --force --options runtime --timestamp --sign "$SIGN_APP" "$STAGE/$APPDIR/RealMotionBlur.plugin"
fi
codesign --verify --strict "$STAGE/$APPDIR/RealMotionBlur.plugin"
xattr -cr "$STAGE"            # no extended attributes, so no ._ files in the payload
export COPYFILE_DISABLE=1

# The plugin must land exactly where the payload says, never "relocated" onto some other copy of the bundle.
pkgbuild --analyze --root "$STAGE" build/pkg/components.plist >/dev/null
# Reinstalling (even the same or an older version) always overwrites.
for i in $(seq 0 $(( $(grep -c "RootRelativeBundlePath" build/pkg/components.plist) - 1 ))); do
  for key in BundleIsRelocatable BundleIsVersionChecked; do
    /usr/libexec/PlistBuddy -c "Delete :$i:$key" build/pkg/components.plist 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :$i:$key bool false" build/pkg/components.plist
  done
done
pkgbuild --root "$STAGE" --install-location / --component-plist build/pkg/components.plist \
  --scripts installer/scripts --identifier $ID --version "$VERSION" build/pkg/component.pkg >/dev/null

sed "s/@VERSION@/$VERSION/g; s/@ID@/$ID/g" installer/distribution.xml > build/pkg/distribution.xml
mkdir -p build/pkg/resources && cp installer/resources/*.html build/pkg/resources/ && cp LICENSE build/pkg/resources/LICENSE.txt
productbuild --distribution build/pkg/distribution.xml --resources build/pkg/resources --package-path build/pkg \
  ${SIGN_PKG:+--sign "$SIGN_PKG"} --timestamp "$OUT" >/dev/null 2>&1 || \
  productbuild --distribution build/pkg/distribution.xml --resources build/pkg/resources --package-path build/pkg \
  ${SIGN_PKG:+--sign "$SIGN_PKG"} "$OUT" >/dev/null

if [[ -n ${NOTARY_PROFILE:-} ]]; then
  [[ -n ${SIGN_APP:-} && -n ${SIGN_PKG:-} ]] || { echo "notarizing needs SIGN_APP and SIGN_PKG"; exit 1; }
  xcrun notarytool submit "$OUT" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$OUT"
  spctl --assess --type install -v "$OUT"
fi
echo "built $OUT"

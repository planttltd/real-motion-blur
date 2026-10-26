#!/bin/zsh
# Removes Real Motion Blur from every After Effects and from /Library/Application Support (asks for your password).
echo "Removing Real Motion Blur..."
sudo zsh -c '
  for ae in /Applications/Adobe\ After\ Effects*/Plug-ins(N/); do rm -rf "$ae/RealMotionBlur.plugin"; done
  rm -rf "/Library/Application Support/Real Motion Blur"
  pkgutil --forget com.plantt.realmotionblur >/dev/null 2>&1
' && echo "Done. Restart After Effects if it is open."

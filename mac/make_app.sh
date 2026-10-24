#!/bin/bash
# Builds "Depth Studio.app" — double-click to start Depth Studio and open it in the
# browser (or just open the browser if it's already running). Quit from the app's
# Quit button. Usage: mac/make_app.sh [destination folder, default ~/Applications]
# bash 3.2 compatible (macOS stock bash).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HOME/Applications}"
APP="$DEST/Depth Studio.app"

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "Run ./setup.sh first (the icon is drawn with the project's Python)."
  exit 1
fi

mkdir -p "$DEST"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- icon
ICONSET="$(mktemp -d)/AppIcon.iconset"
"$REPO/.venv/bin/python" "$REPO/mac/icon.py" "$ICONSET"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"

# --- Info.plist
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Depth Studio</string>
  <key>CFBundleDisplayName</key><string>Depth Studio</string>
  <key>CFBundleIdentifier</key><string>com.irfanabdulhameed.depthstudio</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>DepthStudio</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

# --- launcher (the repo path is baked in; re-run this script if you move the folder)
cat > "$APP/Contents/MacOS/DepthStudio" <<LAUNCHER
#!/bin/bash
REPO="$REPO"
PORT=7860
URL="http://127.0.0.1:\$PORT"
listening() { /usr/sbin/lsof -nP -iTCP:\$PORT -sTCP:LISTEN -t >/dev/null 2>&1; }
alert() { /usr/bin/osascript -e "display alert \"Depth Studio\" message \"\$1\" as warning" >/dev/null; }

if ! listening; then
  if [ ! -x "\$REPO/run.sh" ]; then
    alert "The project folder moved or was deleted (\$REPO). Run mac/make_app.sh again from its new location."
    exit 1
  fi
  if [ ! -x "\$REPO/.venv/bin/python" ] || [ ! -f "\$REPO/vda/checkpoints/video_depth_anything_vits.pth" ]; then
    choice=\$(/usr/bin/osascript -e 'button returned of (display dialog "Depth Studio needs a one-time setup (about 1.4 GB download)." buttons {"Cancel", "Run setup"} default button "Run setup" with title "Depth Studio")' 2>/dev/null)
    if [ "\$choice" = "Run setup" ]; then
      /usr/bin/osascript -e "tell application \"Terminal\" to do script \"cd '\$REPO' && ./setup.sh\"" -e 'tell application "Terminal" to activate' >/dev/null
    fi
    exit 0
  fi
  cd "\$REPO" && nohup ./run.sh >> "\$REPO/run.log" 2>&1 &
  for _ in \$(seq 1 90); do
    listening && break
    sleep 1
  done
  if ! listening; then
    alert "Depth Studio didn't start within 90 seconds. Details are in \$REPO/run.log"
    exit 1
  fi
fi
/usr/bin/open "\$URL"
LAUNCHER
chmod +x "$APP/Contents/MacOS/DepthStudio"
touch "$APP"   # refresh Finder's icon cache for the bundle

echo "✓ built $APP — double-click it to start Depth Studio"

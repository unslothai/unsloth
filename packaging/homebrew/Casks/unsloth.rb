# Canonical Homebrew cask for Unsloth Desktop. Mirror changes to Homebrew/homebrew-cask
# (Casks/u/unsloth.rb) so `brew install --cask unsloth` picks them up.
cask "unsloth" do
  os macos: "MacOS.dmg", linux: "Linux.AppImage"

  version "0.1.908-beta"
  sha256 arm:          "ed66c4e4e1fd3f087b9e3eeba5dcf444dbe7fe42869427aeab5d29d353ba4517",
         x86_64_linux: "37b3736f6cea9bf75bff386fafb3ae3e1572f3767e00706783d3e96ac4156c2b"

  on_macos do
    auto_updates true
    depends_on arch: :arm64

    app "Unsloth.app"

    zap trash: [
      "~/.unsloth/studio",
      "~/Library/Application Support/ai.unsloth.studio",
      "~/Library/Caches/ai.unsloth.studio",
      "~/Library/LaunchAgents/unsloth-studio.plist",
      "~/Library/WebKit/ai.unsloth.studio",
    ]
  end
  on_linux do
    depends_on arch: :x86_64

    app_image "Unsloth-Desktop-Linux.AppImage", target: "Unsloth.AppImage"
    binary "#{appdir}/Unsloth.AppImage", target: "unsloth"

    postflight_steps do
      run "/bin/bash", args: [
        "-c",
        <<~EOS.squish,
          set -euo pipefail
          apps="${HOME}/.local/share/applications"
          mkdir -p "${apps}"
          desktop="${apps}/unsloth-studio.desktop"
          {
            echo "[Desktop Entry]"
            echo "Version=1.0"
            echo "Type=Application"
            echo "Name=Unsloth Desktop"
            echo "Comment=Desktop application for Unsloth Studio"
            echo "Exec=\\"{{appdir}}/Unsloth.AppImage\\" %U"
            echo "Icon={{appdir}}/Unsloth.AppImage"
            echo "Terminal=false"
            echo "Categories=Development;Science;"
            echo "StartupNotify=true"
          } > "${desktop}"
          chmod 755 "${desktop}"
          if command -v update-desktop-database >/dev/null 2>&1; then
            update-desktop-database "${apps}" || true
          fi
        EOS
      ]
    end

    uninstall_postflight_steps do
      run "/bin/bash", args: [
        "-c",
        'rm -f "$HOME/.local/share/applications/unsloth-studio.desktop"',
      ]
    end

    zap trash: [
      "~/.local/share/applications/unsloth-studio.desktop",
    ]
  end

  url "https://github.com/unslothai/unsloth/releases/download/v#{version}/Unsloth-Desktop-#{os}"
  name "Unsloth Desktop"
  desc "Desktop application for Unsloth Studio"
  homepage "https://unsloth.ai/"

  livecheck do
    url :url
    regex(/^v?(\d+(?:\.\d+)+.*)$/i)
    strategy :github_latest
  end
end

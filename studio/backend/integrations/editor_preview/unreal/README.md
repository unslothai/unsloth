# Unreal Engine live preview

Extract the UnslothPreview folder into the Unreal 5.8 project's Plugins directory. Build the project's Editor target using the standard Unreal toolchain, then enable the plugin.

In Studio chat, select **Editor preview → Unreal Engine → Start preview** and copy its pairing code. Open a perspective Level Editor viewport. In the Unreal Output Log console, run **UnslothPreview.Connect** to read the code from the clipboard. Run **UnslothPreview.Disconnect** to stop.

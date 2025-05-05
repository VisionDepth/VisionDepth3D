# VisionDepth3D — Planned Features & Future Roadmap

As VisionDepth3D continues to grow, the focus remains on **usability**, **global accessibility**, and **stronger preview/testing tools**.

---

## 🚀 Planned Updates (Coming Soon)

### 1. 🖼️ Unified Preview Interface Overhaul

* Fix layout bugs in the current Preview GUI
* Fully integrate the **3D Generator Tab** directly into the **VDPlayer interface**
* Enable live preview switching (e.g., Passive Interlaced, Anaglyph, SBS) inside VDPlayer
* Add “before/after” split-view comparison slider
* Display GPU/VRAM usage during preview/render

### 2. 🌐 Multilingual Support

* Introduce a universal language system:

  * Language packs (`en.json`, `fr.json`, `de.json`, etc.)
  * Auto-detect or user-select language from settings
* Add fallback for missing strings
* Create simple GUI or script for translators
* Community contributions encouraged for translation expansion
* Allow language override via CLI

### 3. 💬 Advanced Tooltip Mechanics

* Expand help system beyond basic hover-tooltips:

  * Mouse wheel press or right-click = context-aware help
* Localized tooltips based on selected language
* Beginner mode: show all tooltips permanently
* Tooltip checker to identify missing tooltips

### 4. 🧠 Smarter Model Handling

* Provide basic benchmarking info after model load (speed, memory)
* Detect and display model input/output shapes
* Warn if system VRAM is too low
* Allow model info export to JSON/log

### 5. 🧪 Additional Utilities

* Add batch video/depth processing mode
* Implement frame-by-frame benchmark/testing mode
* Depth visual tuning (blur strength, colormap, edge masking toggle)
* Auto-trim video to depth or vice versa (GUI-friendly)

---

## 📆 Community & Contributions

* Feature requests and translation help welcome via [GitHub Issues](https://github.com/VisionDepth/VisionDepth3D/issues)

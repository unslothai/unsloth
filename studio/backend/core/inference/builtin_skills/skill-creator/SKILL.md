---
name: skill-creator
description: Create or update reusable Agent Skills and install them in Studio. Use when the user asks to make, save, or revise a skill.
metadata:
  author: unslothai
  bundled: "true"
  version: "1.0"
---

# Skill Creator

Create the requested skill only when Code is enabled and Tool permissions is set to Full access. If either is unavailable, tell the user to enable it and stop.

Keep `SKILL.md` focused on guidance that changes the model's decisions. Its YAML frontmatter must contain a lowercase hyphenated `name` matching the folder and a precise `description`. Add text files under `references/`, `scripts/`, or `assets/` only when they are useful.

Use Python through Code to build a ZIP with one skill folder, then install it with Studio's existing validated importer. Locate the backend without assuming an operating system or install path:

```python
from pathlib import Path
import studio
import sys

studio_package = Path(studio.__file__).parent
sys.path.insert(0, str(studio_package / "backend"))

from core.inference.skills import import_skill_archive
from utils.paths import studio_root
```

`import_skill_archive(Path(zip_path), replace=False)` installs and enables the bundle at `studio_root() / "skills" / skill_name`. Pass `replace=True` only when the user explicitly asked to update or overwrite that installed skill.

After installation, report the skill name and tell the user it is available on the next turn. Do not leave temporary ZIP files behind.

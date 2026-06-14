
// @ts-nocheck
import { BUILTIN_PRESETS } from "../../studio/frontend/src/features/chat/presets/preset-policy.ts";
console.log(JSON.stringify({
    names: BUILTIN_PRESETS.map((p) => p.name),
}));

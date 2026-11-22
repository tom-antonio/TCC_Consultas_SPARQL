#!/bin/bash
set -e

TARGET="/app/server/endpoints/workspaceThreads.js"
MARKER="RESET_ON_NEW_THREAD_SENTINEL"

if [ -f "$TARGET" ] && ! grep -q "$MARKER" "$TARGET"; then
  node - "$TARGET" "$MARKER" <<'NODE'
const fs = require("fs");
const [target, marker] = process.argv.slice(2);
let source = fs.readFileSync(target, "utf8");
const needle = '        await Telemetry.sendTelemetry(\n';
const hook = `        // ${marker}\n        try {\n          await fetch("http://backend:8000/api/reset-ontology", { method: "POST" });\n        } catch (error) {\n          console.error("Falha ao resetar a ontologia em nova thread:", error.message);\n        }\n\n`;
if (!source.includes(marker) && source.includes(needle)) {
  source = source.replace(needle, hook + needle);
  fs.writeFileSync(target, source);
  console.log("[startup] New-thread reset hook installed.");
}
NODE
fi

exec /usr/local/bin/docker-entrypoint.sh

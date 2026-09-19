"""Production standalone sample consumer in Chromium, with a stateful API fixture."""
import base64
import subprocess
import json
import os
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
source = (root / "src/extrinsic-legacy.ts").read_text()
ids = ["camera-placeholder", "mode-chip", "input-chip", "error-banner", "success-banner",
       "frame-meta", "coordinate-hint", "result-box", "image-topic", "intrinsic-file", "pose-prefix", "output-file"]
html = '<div style="width:640px;height:480px"><canvas id="camera-canvas" width="640" height="480"></canvas></div>'
html += ''.join('<div id="%s"></div>' % item for item in ids)
html += '<select id="marker-select"></select><table><tbody id="points-body"></tbody></table>'
html += ''.join('<button id="%s-button">%s</button>' % (item,item) for item in ['freeze','live','remove','clear','solve','save'])
red = cv2.imencode('.png', np.full((480,640,3), [0,0,255], dtype=np.uint8))[1].tobytes()
blue = cv2.imencode('.png', np.full((480,640,3), [255,0,0], dtype=np.uint8))[1].tobytes()
# Browser dependency belongs to the existing XGC2 browser-test toolchain. This
# wrapper needs only the already-required OpenCV/NumPy Python runtime.
subprocess.run([os.environ.get("NODE", "node"), str(root / "scripts/check-extrinsic-candidate.mjs")],
               input=json.dumps({"source": source, "html": html,
                                 "red": base64.b64encode(red).decode(), "blue": base64.b64encode(blue).decode()}),
               text=True, check=True)

#!/usr/bin/env python3
"""Extract the documented saved rollouts without executing any notebook cells."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/assets/media-sources.json"


def extract_video(notebook, cell_index, output_index):
    output = notebook["cells"][cell_index]["outputs"][output_index]
    html = output["data"]["text/html"]
    if isinstance(html, list):
        html = "".join(html)
    matches = re.findall(r"data:video/mp4;base64,([A-Za-z0-9+/=\s]+)", html)
    if len(matches) != 1:
        raise ValueError(f"Expected one embedded MP4, found {len(matches)}")
    return base64.b64decode(re.sub(r"\s+", "", matches[0]), validate=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify exports without writing files")
    mode.add_argument("--previews", action="store_true", help="Also create 8-second GIF previews using ffmpeg")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())

    for item in manifest["videos"]:
        source = ROOT / item["notebook"]
        content = extract_video(json.loads(source.read_text()), item["cell_index"], item["output_index"])
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Saved video changed in {source.name}; review its provenance before exporting")
        destination = ROOT / item["mp4"]
        if args.check:
            if destination.read_bytes() != content:
                raise ValueError(f"Export differs from notebook: {destination}")
            print(f"Verified {item['mp4']}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        if args.previews:
            # Preview the opening 8 seconds at the original playback speed.
            # The full MP4 bytes remain exactly as recorded.
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(destination), "-t", "8", "-filter_complex",
                "fps=8,scale=320:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                "-loop", "0", str(ROOT / item["gif"]),
            ], check=True)
        print(f"Exported {item['mp4']}")


if __name__ == "__main__":
    main()

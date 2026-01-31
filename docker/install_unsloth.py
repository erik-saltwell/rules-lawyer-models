#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ref = os.environ.get("UNSLOTH_INSTALLER_REF", "main")
    url = f"https://raw.githubusercontent.com/unslothai/unsloth/{ref}/unsloth/_auto_install.py"
    patched_path = Path("/tmp/_unsloth_auto_install_patched.py")

    print(f"Fetching Unsloth installer: {url}", flush=True)
    raw = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")

    # Patch: make torch.cuda.get_device_capability() safe during docker build (GPU usually not visible).
    helper = (
        textwrap.dedent(
            """
        def _safe_get_device_capability():
            \"\"\"Return (0,0) if CUDA/GPU isn't available (common during docker build).\"\"\"
            try:
                if torch.cuda.is_available():
                    return torch.cuda.get_device_capability()
            except Exception:
                pass
            return (0, 0)
        """
        ).strip()
        + "\n"
    )

    if "_safe_get_device_capability" not in raw:
        m = re.search(r"^\s*import\s+torch\b.*$", raw, flags=re.M)
        if m:
            insert_at = m.end()
            raw = raw[:insert_at] + "\n" + helper + raw[insert_at:]
        else:
            raw = helper + "\n" + raw

    raw = re.sub(r"torch\.cuda\.get_device_capability\(\)", "_safe_get_device_capability()", raw)
    patched_path.write_text(raw, encoding="utf-8")

    proc = subprocess.run([sys.executable, str(patched_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write("--- Unsloth installer STDOUT ---\n")
        sys.stderr.write(proc.stdout + "\n")
        sys.stderr.write("--- Unsloth installer STDERR ---\n")
        sys.stderr.write(proc.stderr + "\n")
        return proc.returncode

    # Find the last install command in stdout.
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    pat = re.compile(r"^(?:\$\s*)?(pip3?|python\s*(?:3)?\s*-m\s+pip|uv\s+pip)\s+install\b")
    candidates = [ln.lstrip("$ ").strip() for ln in lines if pat.match(ln)]

    if not candidates:
        sys.stderr.write("ERROR: Could not find an install command in Unsloth installer output.\n")
        sys.stderr.write("--- Unsloth installer STDOUT ---\n")
        sys.stderr.write(proc.stdout + "\n")
        sys.stderr.write("--- Unsloth installer STDERR ---\n")
        sys.stderr.write(proc.stderr + "\n")
        return 2

    cmdline = candidates[-1]
    if "unsloth" not in cmdline.lower():
        sys.stderr.write("ERROR: Extracted command does not look like an Unsloth install command:\n")
        sys.stderr.write(cmdline + "\n")
        return 3

    print("Extracted install command:", flush=True)
    print(cmdline, flush=True)

    # Execute each segment via `uv pip ...` to avoid requiring the `pip` module in the venv.
    # The upstream command is usually a chain joined by `&&`.
    segments = [seg.strip() for seg in cmdline.split("&&") if seg.strip()]
    for seg in segments:
        # Normalize executable to `uv pip`
        seg = re.sub(r"^(python\s*(?:3)?\s*-m\s+pip|pip3?|uv\s+pip)\s+", "uv pip ", seg.strip())
        argv = shlex.split(seg)

        if len(argv) < 2 or argv[0] != "uv" or argv[1] != "pip":
            raise RuntimeError(f"Unexpected segment (expected uv pip ...): {seg}")

        _run(argv)

    # Smoke test
    _run([sys.executable, "-c", "import unsloth, unsloth_zoo; print('unsloth import OK')"])

    # Cleanup
    try:
        patched_path.unlink()
    except FileNotFoundError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

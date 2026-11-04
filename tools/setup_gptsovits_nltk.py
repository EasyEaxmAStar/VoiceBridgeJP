"""Provision only the installed English frontend's official offline NLTK data."""
from pathlib import Path
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile

import requests

REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
BASE = f"https://raw.githubusercontent.com/nltk/nltk_data/{REVISION}"
ROOT = Path(__file__).resolve().parents[1] / ".model-cache/gptsovits-nltk"
# The legacy zip is probed by g2p-en 2.1.0, never unpickled here. Modern
# nltk.pos_tag actually uses the English JSON tagger.
PACKAGES = {"cmudict": "corpora", "averaged_perceptron_tagger": "taggers",
            "averaged_perceptron_tagger_eng": "taggers"}


def main():
    response = requests.get(BASE + "/index.xml", timeout=30)
    response.raise_for_status()
    index = {p.attrib["id"]: p.attrib for p in ET.fromstring(response.content).findall("packages/package")}
    rows = []
    for name, category in PACKAGES.items():
        metadata = index[name]
        url = f"{BASE}/packages/{category}/{name}.zip"
        target = ROOT / category / (name + ".zip")
        if target.exists():
            content = target.read_bytes()
        else:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != int(metadata["size"]) or digest != metadata["sha256_checksum"]:
            raise RuntimeError(f"Official NLTK archive hash/size mismatch: {name}")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in archive.infolist():
                resolved = (target.parent / member.filename).resolve()
                if not resolved.is_relative_to(target.parent.resolve()):
                    raise ValueError("Unsafe archive path")
                if not member.is_dir() and resolved.exists() and resolved.read_bytes() != archive.read(member):
                    raise ValueError(f"Refusing to overwrite different existing data: {resolved}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(content)
            # No legacy pickle extraction/loading; the compatibility probe only
            # needs the zip. The actual English tagger loads JSON files.
            if name != "averaged_perceptron_tagger":
                for member in archive.infolist():
                    resolved = (target.parent / member.filename).resolve()
                    if member.is_dir():
                        resolved.mkdir(parents=True, exist_ok=True)
                    elif not resolved.exists():
                        resolved.parent.mkdir(parents=True, exist_ok=True)
                        with resolved.open("xb") as stream:
                            stream.write(archive.read(member))
        rows.append({"id": name, "url": url, "bytes": len(content), "sha256": digest,
                     "license": metadata.get("license"), "copyright": metadata.get("copyright")})
        print(name, len(content), digest, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    provenance = ROOT / "provenance.json"
    content = json.dumps({"revision": REVISION, "packages": rows}, indent=2)
    if not provenance.exists():
        with provenance.open("x", encoding="utf8") as stream:
            stream.write(content)
    elif provenance.read_text(encoding="utf8") != content:
        raise ValueError("Refusing to overwrite different existing NLTK provenance")


if __name__ == "__main__":
    main()

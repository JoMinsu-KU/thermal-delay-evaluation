"""Read-only archive checks and sealed, restartable validation outputs."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile

RUN_REL = "runs/L2_diagnostics/diagnosis_20260926/full_relative_s3030"
EPISODE = "l2_090006"
DELAYS = (13.0, 38.0, 50.0)
STEPS = (10.0, 5.0, 2.5)


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    # Exactly matches the original Journal's object_hash serialization.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def under(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), f"Path escapes root: {name}")
    return path


def verify_seal(folder):
    folder = Path(folder)
    require((folder / "COMPLETE").is_file(), f"Incomplete case: {folder}")
    require((folder / "COMPLETE").read_text().strip() == sha256(folder / "manifest.json"), "Manifest changed")
    manifest = read_json(folder / "manifest.json")
    for name, expected in manifest["files"].items():
        require(sha256(under(folder, name)) == expected, f"Changed sealed file: {folder / name}")
    return manifest


def seal(folder, metadata):
    folder = Path(folder)
    require(not (folder / "COMPLETE").exists(), f"Refusing to overwrite sealed case: {folder}")
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name not in ("manifest.json", "COMPLETE")}
    write_json(folder / "manifest.json", {"files": files, "metadata": metadata})
    (folder / "COMPLETE").write_text(sha256(folder / "manifest.json") + "\n", encoding="utf-8")


def read_trace(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_latency(run, cfg):
    """The original loader opens journals rb+; this version never writes them."""
    groups = []
    for repeat in range(cfg["budget"]["latency_repeats"]):
        folder = run / "18_latency/trials" / f"shared_physics_{repeat}"
        manifest = verify_seal(folder)
        identity = {
            "frozen": sha256(run / "15_freeze/frozen.json"),
            "calibration": sha256(run / "16_calibration/calibration.json"),
            "ring": sha256(run / "18_latency/rings/both/ring.json"),
            "method": "shared_physics", "repeat": repeat, "budget": cfg["budget"],
        }
        require(manifest["metadata"] == identity, "Latency trial identity changed")
        expected_identity = object_hash(identity)
        rows = []
        with (folder / "requests.jsonl").open("rb") as stream:
            for line in stream:
                require(line.endswith(b"\n"), "Truncated archived journal; refusing to repair it")
                item = json.loads(line)
                require(item["identity"] == expected_identity and item["index"] == len(rows), "Journal identity/order changed")
                require(item["sha256"] == object_hash(item["row"]), "Journal row corrupted")
                rows.append(item["row"])
        paths = {row["path"] for row in rows if not row.get("error")}
        require(paths == {"normal", "missing_observation", "no_predicted_feasible_candidate"}, f"Unexpected latency path coverage: {paths}")
        groups.append(rows)
    require(len({len(x) for x in groups}) == 1, "Unequal latency trial lengths")
    return [row for group in zip(*groups) for row in group]


def fixed_latency(rows, delay_ms):
    return [dict(row, e2e_ms=delay_ms) if row["fallback_reason"] is None else row for row in rows]


def case_key(step, delay):
    return f"h{step:g}ms_d{delay:g}ms"


def completed_case(output, key, protocol_hash):
    base = Path(output) / "cases" / key
    if not base.exists():
        return None
    sealed = []
    for folder in sorted(base.glob("attempt_*")):
        if (folder / "COMPLETE").exists():
            meta = verify_seal(folder)["metadata"]
            require(meta["protocol_sha256"] == protocol_hash and meta["case"] == key, "Case belongs to another protocol")
            sealed.append(folder)
    require(len(sealed) <= 1, f"Multiple completed attempts for {key}; manual audit needed")
    return sealed[0] if sealed else None


def bundle(output):
    output = Path(output)
    target = output.with_name(output.name + "_results.zip")
    tmp = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(output.parent))
    os.replace(tmp, target)
    return target

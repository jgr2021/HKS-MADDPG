"""Build and verify the ICASSP reproducibility ZIP.

The archive is a scientific snapshot, not a byte-for-byte backup of editor,
cache, temporary-render, or unrelated concurrently written state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import time
import zipfile


PROJECT_ROOT_NAME = "maddpg-pytorch-GSP-v1"
INTERNAL_MANIFEST = "REPRODUCIBLE_ARCHIVE_MANIFEST_SHA256.csv"
CHUNK_SIZE = 8 * 1024 * 1024
STORE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".npz",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".xz",
    ".zip",
}
LATEX_INTERMEDIATES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root (default: directory containing this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/icassp2027_reproducible_project_20260812.zip"),
        help="ZIP path, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read every archived member and verify it against the manifest.",
    )
    return parser.parse_args()


def excluded(relative_path: Path, output_path: Path, root: Path) -> bool:
    parts = relative_path.parts
    lowered = tuple(part.lower() for part in parts)
    if not parts:
        return True
    if relative_path.resolve() == output_path.resolve():
        return True
    if any(part in {".git", ".idea", ".agents", "__pycache__", ".pytest_cache"} for part in lowered):
        return True
    if lowered[0] == "tmp":
        return True
    if lowered[:2] == ("experiments", "support_filter_gate_v1"):
        return True
    if lowered[0] == "maddpg-pytorch-gsp-v1.zip":
        return True
    if lowered[0] == "output" and relative_path.name.lower().startswith(
        "icassp2027_reproducible_project_20260812"
    ):
        return True
    name_lower = relative_path.name.lower()
    if lowered[:2] == ("paper", "icassp_active_gsp") and any(
        name_lower.endswith(suffix) for suffix in LATEX_INTERMEDIATES
    ):
        return True
    return False


def archive_info(path: Path, archive_name: str) -> zipfile.ZipInfo:
    stat = path.stat()
    local_time = time.localtime(stat.st_mtime)
    year = min(max(local_time.tm_year, 1980), 2107)
    info = zipfile.ZipInfo(
        archive_name,
        (year, local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec),
    )
    info.compress_type = (
        zipfile.ZIP_STORED
        if path.suffix.lower() in STORE_SUFFIXES
        else zipfile.ZIP_DEFLATED
    )
    info.external_attr = (stat.st_mode & 0xFFFF) << 16
    return info


def write_streamed_member(
    archive: zipfile.ZipFile, path: Path, archive_name: str
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    info = archive_info(path, archive_name)
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            target.write(chunk)
    return digest.hexdigest().upper(), size


def manifest_bytes(rows: list[tuple[str, int, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["sha256", "size_bytes", "path"])
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_archive(output_path: Path, expected_rows: list[tuple[str, int, str]]) -> None:
    expected = {path: (sha256, size) for sha256, size, path in expected_rows}
    manifest_name = f"{PROJECT_ROOT_NAME}/{INTERNAL_MANIFEST}"
    with zipfile.ZipFile(output_path, "r") as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"CRC verification failed for {bad_crc}")
        with archive.open(manifest_name, "r") as source:
            decoded = io.TextIOWrapper(source, encoding="utf-8", newline="")
            rows = list(csv.DictReader(decoded))
        parsed = {
            row["path"]: (row["sha256"], int(row["size_bytes"])) for row in rows
        }
        if parsed != expected:
            raise RuntimeError("Internal manifest does not match the build manifest")
        for index, (path, (expected_hash, expected_size)) in enumerate(
            sorted(expected.items()), start=1
        ):
            digest = hashlib.sha256()
            size = 0
            with archive.open(f"{PROJECT_ROOT_NAME}/{path}", "r") as source:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
            if size != expected_size or digest.hexdigest().upper() != expected_hash:
                raise RuntimeError(f"Manifest verification failed for {path}")
            if index % 500 == 0 or index == len(expected):
                print(f"verified {index}/{len(expected)} files", flush=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and not excluded(path.relative_to(root), output_path, root)
    ]
    candidates.sort(key=lambda path: path.relative_to(root).as_posix().lower())
    total_bytes = sum(path.stat().st_size for path in candidates)
    print(
        f"archiving {len(candidates)} files ({total_bytes / (1024 ** 3):.3f} GiB) to {output_path}",
        flush=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()
    manifest_rows: list[tuple[str, int, str]] = []
    archived_bytes = 0
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for index, path in enumerate(candidates, start=1):
                relative = path.relative_to(root).as_posix()
                archive_name = str(PurePosixPath(PROJECT_ROOT_NAME) / relative)
                sha256, size = write_streamed_member(archive, path, archive_name)
                manifest_rows.append((sha256, size, relative))
                archived_bytes += size
                if index % 250 == 0 or index == len(candidates):
                    print(
                        f"archived {index}/{len(candidates)} files "
                        f"({archived_bytes / (1024 ** 3):.3f}/{total_bytes / (1024 ** 3):.3f} GiB)",
                        flush=True,
                    )
            manifest = manifest_bytes(manifest_rows)
            archive.writestr(
                f"{PROJECT_ROOT_NAME}/{INTERNAL_MANIFEST}",
                manifest,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    if args.verify:
        verify_archive(output_path, manifest_rows)

    archive_hash = file_sha256(output_path)
    sidecar = output_path.with_suffix(".sha256")
    with sidecar.open("w", encoding="ascii", newline="\n") as target:
        target.write(f"{archive_hash}  {output_path.name}\n")
    print(f"archive_size_bytes={output_path.stat().st_size}", flush=True)
    print(f"archive_sha256={archive_hash}", flush=True)
    print(f"sidecar={sidecar}", flush=True)


if __name__ == "__main__":
    main()

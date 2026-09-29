"""Official MOPTA08 executable adapter used by the GTBO benchmark protocol."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile

import numpy as np


MOPTA08_DIMENSION = 124
MOPTA08_CONSTRAINTS = 68
MOPTA08_PENALTY = 10.0
MOPTA08_ARCHIVE_URL = (
    "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
    "b99e9a5f-7425-46a1-8956-9daa100112c3/content"
)
MOPTA08_ARCHIVE_MD5 = "aa10adbdc3eb119d5da643ba70157c36"
MOPTA08_ARCHIVE_MEMBER = "PC/executable/mopta08.exe"
MOPTA08_EXECUTABLE_SHA256 = (
    "35b1d27ca1e91df646612c2d96bd1cb2e435d7c919d898e6343a2de4e6b25c48"
)
MOPTA08_DATASET_DOI = "https://doi.org/10.7488/ds/7894"


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_mopta08_cache_dir() -> Path:
    configured = os.environ.get("MOPTA08_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / ".cache"
    return (root / "AG_VS" / "mopta08").resolve()


def configured_mopta08_executable() -> Path:
    configured = os.environ.get("MOPTA08_EXECUTABLE")
    if configured:
        return Path(configured).expanduser().resolve()
    return default_mopta08_cache_dir() / "mopta08.exe"


def ensure_mopta08_executable() -> Path:
    """Return a checksum-verified official Windows executable.

    ``MOPTA08_EXECUTABLE`` can point to an existing copy. Otherwise the
    official Edinburgh PC archive is downloaded once into the user cache and
    its published archive checksum and pinned executable checksum are checked.
    """

    executable = configured_mopta08_executable()
    explicitly_configured = bool(os.environ.get("MOPTA08_EXECUTABLE"))
    if executable.is_file():
        actual = _file_digest(executable, "sha256")
        if actual != MOPTA08_EXECUTABLE_SHA256:
            raise RuntimeError(
                f"MOPTA08 executable checksum mismatch at {executable}: "
                f"expected {MOPTA08_EXECUTABLE_SHA256}, got {actual}"
            )
        return executable
    if explicitly_configured:
        raise FileNotFoundError(
            f"MOPTA08_EXECUTABLE does not exist: {executable}"
        )

    executable.parent.mkdir(parents=True, exist_ok=True)
    archive = executable.parent / "PC.zip"
    if not archive.is_file() or _file_digest(archive, "md5") != MOPTA08_ARCHIVE_MD5:
        partial = executable.parent / f"PC.zip.part-{os.getpid()}"
        try:
            with urlopen(MOPTA08_ARCHIVE_URL, timeout=120) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            actual = _file_digest(partial, "md5")
            if actual != MOPTA08_ARCHIVE_MD5:
                raise RuntimeError(
                    "Official MOPTA08 archive checksum mismatch: "
                    f"expected {MOPTA08_ARCHIVE_MD5}, got {actual}"
                )
            os.replace(partial, archive)
        finally:
            partial.unlink(missing_ok=True)

    extracted = executable.parent / f"mopta08.exe.part-{os.getpid()}"
    try:
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(MOPTA08_ARCHIVE_MEMBER) as source:
                with extracted.open("wb") as target:
                    shutil.copyfileobj(source, target)
        actual = _file_digest(extracted, "sha256")
        if actual != MOPTA08_EXECUTABLE_SHA256:
            raise RuntimeError(
                "Extracted MOPTA08 executable checksum mismatch: "
                f"expected {MOPTA08_EXECUTABLE_SHA256}, got {actual}"
            )
        try:
            os.replace(extracted, executable)
        except FileExistsError:
            pass
    finally:
        extracted.unlink(missing_ok=True)
    return executable


def penalized_mopta08_value(output: object) -> float:
    """Convert raw objective plus 68 constraints to the runner's maximization."""

    values = np.asarray(output, dtype=np.float64).reshape(-1)
    expected = 1 + MOPTA08_CONSTRAINTS
    if values.size != expected:
        raise RuntimeError(
            f"MOPTA08 produced {values.size} values; expected {expected}"
        )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("MOPTA08 produced non-finite output")
    minimized = float(values[0] + MOPTA08_PENALTY * np.maximum(values[1:], 0).sum())
    return -minimized


class Mopta08Objective:
    """Evaluate the official 124-D soft-constrained MOPTA08 objective."""

    def __init__(self, executable: object | None = None):
        self.executable = (
            Path(executable).expanduser().resolve()
            if executable is not None
            else ensure_mopta08_executable()
        )
        if not self.executable.is_file():
            raise FileNotFoundError(f"MOPTA08 executable does not exist: {self.executable}")
        self._directory = tempfile.TemporaryDirectory(prefix="agvs_mopta08_")
        self._closed = False

    def __call__(self, x: object) -> float:
        if self._closed:
            raise RuntimeError("MOPTA08 objective is already closed")
        candidate = np.asarray(x, dtype=np.float64).reshape(-1)
        if candidate.size != MOPTA08_DIMENSION:
            raise ValueError(
                f"MOPTA08 candidate has {candidate.size} values; "
                f"expected {MOPTA08_DIMENSION}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError("MOPTA08 candidate must contain only finite values")
        if np.any(candidate < 0.0) or np.any(candidate > 1.0):
            raise ValueError("MOPTA08 candidate must lie in [0, 1]^124")

        workdir = Path(self._directory.name)
        input_path = workdir / "input.txt"
        output_path = workdir / "output.txt"
        output_path.unlink(missing_ok=True)
        input_path.write_text(
            "".join(f"{value:.17g}\n" for value in candidate),
            encoding="ascii",
        )
        completed = subprocess.run(
            [str(self.executable)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MOPTA08 executable exited with code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        if not output_path.is_file():
            raise RuntimeError("MOPTA08 executable did not create output.txt")
        try:
            values = [
                float(line.strip())
                for line in output_path.read_text(encoding="ascii").splitlines()
                if line.strip()
            ]
        except ValueError as exc:
            raise RuntimeError("MOPTA08 output.txt contains a non-numeric value") from exc
        return penalized_mopta08_value(values)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._directory.cleanup()


def preflight_mopta08_backend() -> dict[str, object]:
    executable = ensure_mopta08_executable()
    return {
        "task": "mopta08",
        "backend": "official_mopta08_windows_executable",
        "physical_dim": MOPTA08_DIMENSION,
        "executable": str(executable),
        "executable_sha256": MOPTA08_EXECUTABLE_SHA256,
        "dataset_doi": MOPTA08_DATASET_DOI,
        "objective": "maximize -(mass + 10 * sum(max(0, constraints)))",
    }


__all__ = [
    "MOPTA08_CONSTRAINTS",
    "MOPTA08_DATASET_DOI",
    "MOPTA08_DIMENSION",
    "MOPTA08_EXECUTABLE_SHA256",
    "Mopta08Objective",
    "configured_mopta08_executable",
    "ensure_mopta08_executable",
    "penalized_mopta08_value",
    "preflight_mopta08_backend",
]

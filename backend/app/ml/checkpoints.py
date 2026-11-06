"""Transactional training backups with integrity validation and bounded retention."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


class CheckpointError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BackupManager:
    def __init__(self, root: Path, run_id: str, max_backups: int = 2):
        if max_backups < 1:
            raise ValueError("max_backups must be positive")
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.directory = (self.root / run_id).resolve()
        if self.directory.parent != self.root:
            raise ValueError("run_id must identify a direct backup directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.counter_path = self.directory / "counter.json"
        self.max_backups = max_backups

    def _next_number(self) -> int:
        current = 0
        if self.counter_path.exists():
            try:
                current = int(json.loads(self.counter_path.read_text())["next"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CheckpointError("Backup counter is invalid; refusing to reuse numbers") from exc
        numbers = [
            int(path.name.rsplit("-", 1)[1])
            for path in self.directory.iterdir()
            if path.is_dir() and path.name.startswith("backup-train-")
            and path.name.rsplit("-", 1)[1].isdigit()
        ]
        number = max([current, *numbers], default=0) + 1
        temporary = self.counter_path.with_name(f".{self.counter_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps({"next": number}))
        os.replace(temporary, self.counter_path)
        return number

    @staticmethod
    def _manifest(directory: Path) -> dict:
        files = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files[str(path.relative_to(directory))] = {
                    "size": path.stat().st_size,
                    "sha256": _digest(path),
                }
        return files

    def validate(self, directory: Path) -> bool:
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            files = manifest["files"]
            return all(
                (directory / name).is_file()
                and (directory / name).stat().st_size == info["size"]
                and _digest(directory / name) == info["sha256"]
                for name, info in files.items()
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def save(self, writer) -> Path:
        number = self._next_number()
        temporary = Path(tempfile.mkdtemp(prefix=".backup-", dir=self.directory))
        target = self.directory / f"backup-train-{number:02d}"
        try:
            writer(temporary)
            files = self._manifest(temporary)
            if not files:
                raise CheckpointError("Backup contains no files")
            (temporary / "manifest.json").write_text(json.dumps(
                {"schema_version": 1, "run_id": self.run_id, "number": number, "files": files},
                indent=2, sort_keys=True,
            ))
            if not self.validate(temporary):
                raise CheckpointError("Backup integrity validation failed")
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._retain()
        return target

    def valid_backups(self) -> list[Path]:
        candidates = sorted(
            (path for path in self.directory.iterdir()
             if path.is_dir() and path.name.startswith("backup-train-")),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        return [path for path in candidates if self.validate(path)]

    def latest(self) -> Path:
        backups = self.valid_backups()
        if not backups:
            raise CheckpointError("No valid training backup found")
        return backups[-1]

    def _retain(self):
        valid = self.valid_backups()
        for path in valid[:-self.max_backups]:
            if path.is_symlink() or path.parent != self.directory:
                continue
            shutil.rmtree(path)

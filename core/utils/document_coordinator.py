"""
Document coordination utilities for preventing contention between multiple instances.
"""

import fcntl
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DocumentStatus:
    """Status information for a document."""

    file_path: str
    file_hash: str
    status: str  # 'processing', 'completed', 'failed'
    instance_id: str
    started_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class DocumentCoordinator:
    """Coordinates document processing across multiple instances to prevent contention."""

    def __init__(self, coordination_dir: Path, instance_id: str):
        self.coordination_dir = Path(coordination_dir)
        self.instance_id = instance_id
        self.documents_registry_file = self.coordination_dir / "documents.json"
        self.lock_dir = self.coordination_dir / "locks"

        # Create directories if they don't exist
        self.coordination_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        # Initialize documents registry
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize the documents registry file."""
        if not self.documents_registry_file.exists():
            with open(self.documents_registry_file, "w") as f:
                json.dump({}, f)

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for the file based on path and modification time."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Use file path and modification time for hash
        stat = file_path.stat()
        content = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_lock_file_path(self, file_path: Path) -> Path:
        """Get the lock file path for a given document."""
        file_hash = self._get_file_hash(file_path)
        return self.lock_dir / f"{file_hash}.lock"

    def _load_registry(self) -> Dict[str, DocumentStatus]:
        """Load the documents registry from file."""
        try:
            with open(self.documents_registry_file, "r") as f:
                data = json.load(f)

            # Convert dict back to DocumentStatus objects
            registry = {}
            for file_hash, status_dict in data.items():
                registry[file_hash] = DocumentStatus(**status_dict)

            return registry
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_registry(self, registry: Dict[str, DocumentStatus]):
        """Save the documents registry to file."""
        # Convert DocumentStatus objects to dicts
        data = {}
        for file_hash, status in registry.items():
            data[file_hash] = asdict(status)

        with open(self.documents_registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def _lock_registry(self):
        """Context manager for locking the registry file."""

        class RegistryLock:
            def __init__(self, registry_file):
                self.registry_file = registry_file
                self.lock_file = None

            def __enter__(self):
                self.lock_file = open(f"{self.registry_file}.lock", "w")
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.lock_file:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                    self.lock_file.close()

        return RegistryLock(self.documents_registry_file)

    def can_process_file(self, file_path: Path) -> bool:
        """Check if a file can be processed by this instance."""
        try:
            file_hash = self._get_file_hash(file_path)

            with self._lock_registry():
                registry = self._load_registry()

                # Check if file is in registry
                if file_hash not in registry:
                    return True  # New file, can process

                status = registry[file_hash]

                # If completed successfully, skip
                if status.status == "completed":
                    logger.debug(f"File {file_path} already completed by {status.instance_id}")
                    return False

                # If currently being processed by another instance
                if status.status == "processing" and status.instance_id != self.instance_id:
                    # Check if the processing instance is still active (timeout check)
                    started_at = datetime.fromisoformat(status.started_at)
                    if datetime.now() - started_at < timedelta(hours=1):  # 1 hour timeout
                        logger.debug(f"File {file_path} is being processed by {status.instance_id}")
                        return False
                    else:
                        logger.warning(
                            f"File {file_path} processing by {status.instance_id} timed out, allowing reprocessing"
                        )
                        return True

                # If failed and retry count is below limit
                if status.status == "failed" and status.retry_count < 3:
                    logger.info(f"File {file_path} failed previously, allowing retry ({status.retry_count + 1}/3)")
                    return True

                # If failed too many times
                if status.status == "failed" and status.retry_count >= 3:
                    logger.warning(f"File {file_path} failed too many times, skipping")
                    return False

                # Default to allowing processing for same instance
                return status.instance_id == self.instance_id

        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return True  # Allow processing on error

    def acquire_file_lock(self, file_path: Path) -> bool:
        """Acquire a lock on a file for processing."""
        try:
            file_hash = self._get_file_hash(file_path)
            lock_file_path = self._get_lock_file_path(file_path)

            # Try to create lock file
            if lock_file_path.exists():
                # Check if lock is stale (older than 1 hour)
                lock_age = time.time() - lock_file_path.stat().st_mtime
                if lock_age > 3600:  # 1 hour
                    logger.warning(f"Removing stale lock for {file_path}")
                    lock_file_path.unlink()
                else:
                    logger.debug(f"File {file_path} is locked by another instance")
                    return False

            # Create lock file
            with open(lock_file_path, "w") as f:
                f.write(f"{self.instance_id}:{datetime.now().isoformat()}")

            # Update registry
            with self._lock_registry():
                registry = self._load_registry()

                # Get current status if exists
                current_status = registry.get(file_hash)
                retry_count = (
                    current_status.retry_count + 1 if current_status and current_status.status == "failed" else 0
                )

                registry[file_hash] = DocumentStatus(
                    file_path=str(file_path),
                    file_hash=file_hash,
                    status="processing",
                    instance_id=self.instance_id,
                    started_at=datetime.now().isoformat(),
                    retry_count=retry_count,
                )
                self._save_registry(registry)

            logger.debug(f"Acquired lock for file {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error acquiring lock for {file_path}: {e}")
            return False

    def release_file_lock(self, file_path: Path):
        """Release the lock on a file."""
        try:
            lock_file_path = self._get_lock_file_path(file_path)
            if lock_file_path.exists():
                lock_file_path.unlink()
                logger.debug(f"Released lock for file {file_path}")

        except Exception as e:
            logger.error(f"Error releasing lock for {file_path}: {e}")

    def mark_file_completed(self, file_path: Path):
        """Mark a file as completed successfully."""
        try:
            file_hash = self._get_file_hash(file_path)

            with self._lock_registry():
                registry = self._load_registry()

                if file_hash in registry:
                    registry[file_hash].status = "completed"
                    registry[file_hash].completed_at = datetime.now().isoformat()
                    self._save_registry(registry)

                    logger.debug(f"Marked file {file_path} as completed")

        except Exception as e:
            logger.error(f"Error marking file {file_path} as completed: {e}")

    def mark_file_failed(self, file_path: Path, error_message: str):
        """Mark a file as failed."""
        try:
            file_hash = self._get_file_hash(file_path)

            with self._lock_registry():
                registry = self._load_registry()

                if file_hash in registry:
                    registry[file_hash].status = "failed"
                    registry[file_hash].completed_at = datetime.now().isoformat()
                    registry[file_hash].error_message = error_message
                    self._save_registry(registry)

                    logger.warning(f"Marked file {file_path} as failed: {error_message}")

        except Exception as e:
            logger.error(f"Error marking file {file_path} as failed: {e}")

    def get_processing_status(self) -> Dict[str, int]:
        """Get the current processing status summary."""
        try:
            with self._lock_registry():
                registry = self._load_registry()

                status_counts = {"total": len(registry), "processing": 0, "completed": 0, "failed": 0}

                for status in registry.values():
                    if status.status in status_counts:
                        status_counts[status.status] += 1

                return status_counts

        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {"total": 0, "processing": 0, "completed": 0, "failed": 0}

    def cleanup_stale_locks(self, max_age_hours: int = 1):
        """Clean up stale lock files."""
        try:
            current_time = time.time()
            stale_threshold = max_age_hours * 3600

            for lock_file in self.lock_dir.glob("*.lock"):
                lock_age = current_time - lock_file.stat().st_mtime
                if lock_age > stale_threshold:
                    logger.info(f"Removing stale lock file: {lock_file}")
                    lock_file.unlink()

        except Exception as e:
            logger.error(f"Error cleaning up stale locks: {e}")

    def reset_failed_files(self):
        """Reset all failed files to allow reprocessing."""
        try:
            with self._lock_registry():
                registry = self._load_registry()

                reset_count = 0
                for file_hash, status in registry.items():
                    if status.status == "failed":
                        del registry[file_hash]
                        reset_count += 1

                self._save_registry(registry)
                logger.info(f"Reset {reset_count} failed files for reprocessing")

        except Exception as e:
            logger.error(f"Error resetting failed files: {e}")

    def get_files_by_status(self, status: str) -> List[DocumentStatus]:
        """Get all files with a specific status."""
        try:
            with self._lock_registry():
                registry = self._load_registry()
                return [doc_status for doc_status in registry.values() if doc_status.status == status]

        except Exception as e:
            logger.error(f"Error getting files by status {status}: {e}")
            return []

"""
Instance coordination utilities for safe parallel execution of multiple nBedR instances.
"""

import fcntl
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class InstanceInfo:
    """Information about a running nBedR instance."""

    instance_id: str
    process_id: int
    started_at: str
    config_hash: str
    output_path: str
    vector_db_path: Optional[str] = None
    status: str = "running"
    heartbeat: str = ""


class InstanceCoordinator:
    """Coordinates multiple nBedR instances to prevent conflicts."""

    def __init__(self, config_hash: str, output_path: str, vector_db_path: Optional[str] = None):
        """Initialize instance coordinator.

        Args:
            config_hash: Hash of the configuration to identify compatible instances
            output_path: Output directory path
            vector_db_path: Vector database path (if applicable)
        """
        self.instance_id = str(uuid.uuid4())
        self.config_hash = config_hash
        self.output_path = str(Path(output_path).resolve())
        self.vector_db_path = str(Path(vector_db_path).resolve()) if vector_db_path else None

        # Create coordination directory
        self.coordination_dir = Path(tempfile.gettempdir()) / "nbedr_coordination"
        self.coordination_dir.mkdir(exist_ok=True)

        # Instance registry file
        self.registry_file = self.coordination_dir / "instances.json"
        self.lock_file = self.coordination_dir / "registry.lock"

        logger.info(f"Instance coordinator initialized: {self.instance_id}")

    def register_instance(self) -> bool:
        """Register this instance in the coordination registry.

        Returns:
            True if registration successful, False if conflicts detected
        """
        instance_info = InstanceInfo(
            instance_id=self.instance_id,
            process_id=os.getpid(),
            started_at=datetime.now().isoformat(),
            config_hash=self.config_hash,
            output_path=self.output_path,
            vector_db_path=self.vector_db_path,
            heartbeat=datetime.now().isoformat(),
        )

        with self._lock_registry():
            registry = self._load_registry()

            # Check for conflicts
            conflicts = self._check_conflicts(registry, instance_info)
            if conflicts:
                logger.error(f"Instance conflicts detected: {conflicts}")
                return False

            # Register this instance
            registry[self.instance_id] = asdict(instance_info)
            self._save_registry(registry)

        logger.info(f"Instance {self.instance_id} registered successfully")
        return True

    def unregister_instance(self):
        """Unregister this instance from the coordination registry."""
        with self._lock_registry():
            registry = self._load_registry()
            registry.pop(self.instance_id, None)
            self._save_registry(registry)

        logger.info(f"Instance {self.instance_id} unregistered")

    def update_heartbeat(self):
        """Update heartbeat to indicate this instance is still active."""
        with self._lock_registry():
            registry = self._load_registry()
            if self.instance_id in registry:
                registry[self.instance_id]["heartbeat"] = datetime.now().isoformat()
                registry[self.instance_id]["status"] = "running"
                self._save_registry(registry)

    def get_active_instances(self) -> List[InstanceInfo]:
        """Get list of active instances."""
        with self._lock_registry():
            registry = self._load_registry()
            self._cleanup_stale_instances(registry)

            active_instances = []
            for instance_data in registry.values():
                instance_info = InstanceInfo(**instance_data)
                active_instances.append(instance_info)

            return active_instances

    def get_compatible_instances(self) -> List[InstanceInfo]:
        """Get instances with compatible configuration."""
        all_instances = self.get_active_instances()
        return [inst for inst in all_instances if inst.config_hash == self.config_hash]

    def suggest_instance_specific_paths(self) -> Dict[str, str]:
        """Suggest instance-specific paths to avoid conflicts.

        Returns:
            Dictionary with suggested paths for this instance
        """
        base_output = Path(self.output_path)
        base_name = base_output.name
        parent_dir = base_output.parent

        # Create instance-specific paths
        instance_suffix = self.instance_id[:8]

        suggested_paths = {
            "output_path": str(parent_dir / f"{base_name}_{instance_suffix}"),
            "faiss_index_path": str(parent_dir / f"faiss_index_{instance_suffix}"),
            "temp_path": str(parent_dir / f"temp_{instance_suffix}"),
            "logs_path": str(parent_dir / f"logs_{instance_suffix}"),
        }

        if self.vector_db_path:
            vector_base = Path(self.vector_db_path)
            suggested_paths["vector_db_path"] = str(vector_base.parent / f"{vector_base.name}_{instance_suffix}")

        return suggested_paths

    @contextmanager
    def exclusive_file_access(self, file_path: str, mode: str = "r"):
        """Context manager for exclusive file access with locking.

        Args:
            file_path: Path to the file
            mode: File open mode
        """
        lock_path = f"{file_path}.lock"

        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                logger.debug(f"Acquired exclusive lock for {file_path}")

                with open(file_path, mode) as f:
                    yield f

        except Exception as e:
            logger.error(f"Failed to acquire exclusive access to {file_path}: {e}")
            raise
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass

    def _check_conflicts(self, registry: Dict[str, Any], new_instance: InstanceInfo) -> List[str]:
        """Check for conflicts with existing instances."""
        conflicts = []

        for instance_id, instance_data in registry.items():
            if instance_id == self.instance_id:
                continue

            existing = InstanceInfo(**instance_data)

            # Check for same output path
            if existing.output_path == new_instance.output_path:
                conflicts.append(f"Output path conflict with instance {instance_id}")

            # Check for same vector database path
            if (
                existing.vector_db_path
                and new_instance.vector_db_path
                and existing.vector_db_path == new_instance.vector_db_path
            ):
                conflicts.append(f"Vector database path conflict with instance {instance_id}")

        return conflicts

    def _cleanup_stale_instances(self, registry: Dict[str, Any]):
        """Remove stale instances from registry."""
        current_time = datetime.now()
        stale_instances = []

        for instance_id, instance_data in registry.items():
            try:
                heartbeat_time = datetime.fromisoformat(instance_data["heartbeat"])
                time_diff = (current_time - heartbeat_time).total_seconds()

                # Consider instance stale if no heartbeat for 5 minutes
                if time_diff > 300:
                    stale_instances.append(instance_id)
                    logger.info(f"Marking instance {instance_id} as stale (no heartbeat for {time_diff:.0f}s)")
            except (KeyError, ValueError):
                # Invalid heartbeat data
                stale_instances.append(instance_id)

        for instance_id in stale_instances:
            registry.pop(instance_id, None)

        if stale_instances:
            self._save_registry(registry)

    @contextmanager
    def _lock_registry(self):
        """Context manager for registry file locking."""
        try:
            with open(self.lock_file, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
        except Exception as e:
            logger.error(f"Failed to acquire registry lock: {e}")
            raise
        finally:
            try:
                os.unlink(self.lock_file)
            except FileNotFoundError:
                pass

    def _load_registry(self) -> Dict[str, Any]:
        """Load instance registry from file."""
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file, "r") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load registry, starting with empty: {e}")
            return {}

    def _save_registry(self, registry: Dict[str, Any]):
        """Save instance registry to file."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")
            raise


class SharedRateLimiter:
    """Rate limiter that coordinates across multiple instances."""

    def __init__(self, coordinator: InstanceCoordinator, base_rate_limit: int):
        """Initialize shared rate limiter.

        Args:
            coordinator: Instance coordinator
            base_rate_limit: Base rate limit (will be divided among instances)
        """
        self.coordinator = coordinator
        self.base_rate_limit = base_rate_limit
        self.state_file = coordinator.coordination_dir / f"rate_limit_{coordinator.config_hash}.json"

    def get_instance_rate_limit(self) -> int:
        """Get rate limit for this instance based on number of active instances."""
        compatible_instances = self.coordinator.get_compatible_instances()
        num_instances = len(compatible_instances)

        if num_instances == 0:
            return self.base_rate_limit

        # Divide rate limit among compatible instances
        per_instance_limit = max(1, self.base_rate_limit // num_instances)

        logger.debug(
            f"Rate limit for instance {self.coordinator.instance_id}: "
            f"{per_instance_limit} RPM (total: {self.base_rate_limit}, instances: {num_instances})"
        )

        return per_instance_limit

    def update_shared_state(self, tokens_used: int, response_time: float):
        """Update shared rate limiting state."""
        state_data = {
            "instance_id": self.coordinator.instance_id,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": tokens_used,
            "response_time": response_time,
        }

        with self.coordinator._lock_registry():
            # Load existing shared state
            shared_state = self._load_shared_state()

            # Add this instance's data
            if "instances" not in shared_state:
                shared_state["instances"] = {}

            shared_state["instances"][self.coordinator.instance_id] = state_data

            # Cleanup old data (keep last hour only)
            self._cleanup_old_state(shared_state)

            # Save updated state
            self._save_shared_state(shared_state)

    def _load_shared_state(self) -> Dict[str, Any]:
        """Load shared rate limiting state."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, "r") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_shared_state(self, state: Dict[str, Any]):
        """Save shared rate limiting state."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save shared rate limiting state: {e}")

    def _cleanup_old_state(self, state: Dict[str, Any]):
        """Remove rate limiting data older than 1 hour."""
        cutoff_time = datetime.now().timestamp() - 3600  # 1 hour ago

        if "instances" in state:
            instances_to_remove = []
            for instance_id, instance_data in state["instances"].items():
                try:
                    timestamp = datetime.fromisoformat(instance_data["timestamp"]).timestamp()
                    if timestamp < cutoff_time:
                        instances_to_remove.append(instance_id)
                except (KeyError, ValueError):
                    instances_to_remove.append(instance_id)

            for instance_id in instances_to_remove:
                state["instances"].pop(instance_id, None)


def create_instance_coordinator(config) -> InstanceCoordinator:
    """Create instance coordinator from configuration.

    Args:
        config: EmbeddingConfig instance

    Returns:
        Configured InstanceCoordinator
    """
    # Create configuration hash for compatibility checking
    config_hash = str(hash(f"{config.embedding_provider}_{config.vector_db_type}_{config.embedding_model}"))

    # Determine vector database path
    vector_db_path = None
    if config.vector_db_type == "faiss":
        vector_db_path = config.faiss_index_path
    elif config.vector_db_type == "pgvector":
        vector_db_path = f"{config.pgvector_host}:{config.pgvector_port}/{config.pgvector_database}"

    return InstanceCoordinator(config_hash=config_hash, output_path=config.output, vector_db_path=vector_db_path)

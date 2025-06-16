"""
Tests for document coordination to prevent contention between multiple instances.
"""

import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from filelock import FileLock

from nbedr.core.utils.document_coordinator import DocumentCoordinator, DocumentStatus


class TestDocumentCoordinator:
    """Test cases for DocumentCoordinator."""

    @pytest.fixture
    def temp_coordination_dir(self):
        """Create a unique temporary coordination directory per test run."""
        import os
        import sys

        # Respect TMPDIR from CI environment for parallel builds
        base_tmpdir = os.environ.get('TMPDIR')
        if base_tmpdir:
            # Use CI-provided TMPDIR (e.g., tmp/coord_ubuntu-latest_py3.11)
            base_path = Path(base_tmpdir)
            base_path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=base_path,
                prefix=f"test_{os.getpid()}_"
            ) as temp_dir:
                yield Path(temp_dir)
        else:
            # Fallback for local development
            with tempfile.TemporaryDirectory(
                prefix=f"coord_py{sys.version_info.major}{sys.version_info.minor}_{os.getpid()}_"
            ) as temp_dir:
                yield Path(temp_dir)

    @pytest.fixture
    def coordinator(self, temp_coordination_dir):
        """Create a DocumentCoordinator instance."""
        return DocumentCoordinator(temp_coordination_dir, "test-instance-1")

    @pytest.fixture
    def test_file(self):
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for document coordination")
            test_file_path = Path(f.name)

        yield test_file_path

        # Cleanup
        if test_file_path.exists():
            test_file_path.unlink()

    def test_initialization(self, temp_coordination_dir):
        """Test coordinator initialization."""
        coordinator = DocumentCoordinator(temp_coordination_dir, "test-instance")

        assert coordinator.coordination_dir == temp_coordination_dir
        assert coordinator.instance_id == "test-instance"
        assert coordinator.documents_registry_file.exists()
        assert coordinator.lock_dir.exists()

    def test_file_hash_generation(self, coordinator, test_file):
        """Test file hash generation."""
        hash1 = coordinator._get_file_hash(test_file)
        hash2 = coordinator._get_file_hash(test_file)

        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 characters

    def test_can_process_new_file(self, coordinator, test_file):
        """Test processing new file (not in registry)."""
        assert coordinator.can_process_file(test_file) is True

    def test_acquire_and_release_file_lock(self, coordinator, test_file):
        """Test acquiring and releasing file locks."""
        # Should be able to acquire lock
        assert coordinator.acquire_file_lock(test_file) is True

        # Lock file should exist
        lock_file = coordinator._get_lock_file_path(test_file)
        assert lock_file.exists()

        # Release lock
        coordinator.release_file_lock(test_file)
        assert not lock_file.exists()

    def test_concurrent_lock_acquisition(self, temp_coordination_dir, test_file):
        """Test that only one instance can acquire a lock."""
        coordinator1 = DocumentCoordinator(temp_coordination_dir, "instance-1")
        coordinator2 = DocumentCoordinator(temp_coordination_dir, "instance-2")

        # First instance acquires lock
        assert coordinator1.acquire_file_lock(test_file) is True

        # Second instance should not be able to acquire lock
        assert coordinator2.acquire_file_lock(test_file) is False

        # After first instance releases lock, second should be able to acquire
        coordinator1.release_file_lock(test_file)
        assert coordinator2.acquire_file_lock(test_file) is True

        # Cleanup
        coordinator2.release_file_lock(test_file)

    def test_mark_file_completed(self, coordinator, test_file):
        """Test marking file as completed."""
        # Acquire lock first
        coordinator.acquire_file_lock(test_file)

        # Mark as completed
        coordinator.mark_file_completed(test_file)

        # Check registry
        registry = coordinator._load_registry()
        file_hash = coordinator._get_file_hash(test_file)

        assert file_hash in registry
        assert registry[file_hash].status == "completed"
        assert registry[file_hash].completed_at is not None

        # Should not be able to process completed file
        assert coordinator.can_process_file(test_file) is False

        # Cleanup
        coordinator.release_file_lock(test_file)

    def test_mark_file_failed_with_retry(self, coordinator, test_file):
        """Test marking file as failed and retry logic."""
        # First failure - retry_count will be 0 (initial)
        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Test error")
        coordinator.release_file_lock(test_file)

        # Should still be able to process (retry_count=0, limit is < 3)
        assert coordinator.can_process_file(test_file) is True

        # Second failure - retry_count will be 1 after acquire_file_lock
        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Test error 2")
        coordinator.release_file_lock(test_file)

        # Should still be able to process (retry_count=1, limit is < 3)
        assert coordinator.can_process_file(test_file) is True

        # Third failure - retry_count will be 2 after acquire_file_lock
        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Test error 3")
        coordinator.release_file_lock(test_file)

        # Should still be able to process (retry_count=2, still < 3)
        assert coordinator.can_process_file(test_file) is True

        # Fourth failure - retry_count will be 3 after acquire_file_lock
        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Test error 4")
        coordinator.release_file_lock(test_file)

        # Now should not be able to process (retry_count=3, >= 3)
        assert coordinator.can_process_file(test_file) is False

    def test_processing_status_summary(self, coordinator, test_file):
        """Test getting processing status summary."""
        # Initially empty
        status = coordinator.get_processing_status()
        assert status["total"] == 0

        # Add a processing file
        coordinator.acquire_file_lock(test_file)
        status = coordinator.get_processing_status()
        assert status["total"] == 1
        assert status["processing"] == 1

        # Mark as completed
        coordinator.mark_file_completed(test_file)
        status = coordinator.get_processing_status()
        assert status["completed"] == 1

        coordinator.release_file_lock(test_file)

    def test_cleanup_stale_locks(self, coordinator, test_file):
        """Test cleanup of stale lock files."""
        # Create a stale lock file
        lock_file = coordinator._get_lock_file_path(test_file)
        lock_file.touch()

        # Make it appear old by modifying mtime using os.utime
        import os

        old_time = time.time() - 7200  # 2 hours ago
        os.utime(lock_file, (old_time, old_time))

        assert lock_file.exists()

        # Cleanup stale locks (1 hour threshold)
        coordinator.cleanup_stale_locks(max_age_hours=1)

        # Lock file should be removed
        assert not lock_file.exists()

    def test_reset_failed_files(self, coordinator, test_file):
        """Test resetting failed files for reprocessing."""
        # Mark file as failed multiple times to exceed retry limit (need 4 failures to get retry_count=3)
        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Error 1")
        coordinator.release_file_lock(test_file)

        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Error 2")
        coordinator.release_file_lock(test_file)

        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Error 3")
        coordinator.release_file_lock(test_file)

        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_failed(test_file, "Error 4")
        coordinator.release_file_lock(test_file)

        # Verify file is not processable due to retry limit (retry_count >= 3)
        assert coordinator.can_process_file(test_file) is False

        # Reset failed files
        coordinator.reset_failed_files()

        # Should now be processable (file removed from registry)
        assert coordinator.can_process_file(test_file) is True

    def test_get_files_by_status(self, coordinator, test_file):
        """Test getting files by status."""
        # No files initially
        completed_files = coordinator.get_files_by_status("completed")
        assert len(completed_files) == 0

        # Add and complete a file
        coordinator.acquire_file_lock(test_file)
        coordinator.mark_file_completed(test_file)
        coordinator.release_file_lock(test_file)

        # Should find the completed file
        completed_files = coordinator.get_files_by_status("completed")
        assert len(completed_files) == 1
        assert completed_files[0].status == "completed"
        assert completed_files[0].file_path == str(test_file)


class TestMultipleInstanceScenarios:
    """Test scenarios with multiple instances."""

    @pytest.fixture
    def temp_coordination_dir(self):
        """Create a temporary coordination directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def test_files(self):
        """Create multiple test files."""
        files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(mode="w", suffix=f"_test_{i}.txt", delete=False) as f:
                f.write(f"Test content for file {i}")
                files.append(Path(f.name))

        yield files

        # Cleanup
        for file_path in files:
            if file_path.exists():
                file_path.unlink()

    def test_concurrent_processing_different_files(self, temp_coordination_dir, test_files):
        """Test multiple instances processing different files concurrently."""

        def process_file(instance_id, file_path):
            coordinator = DocumentCoordinator(temp_coordination_dir, instance_id)

            if coordinator.can_process_file(file_path):
                if coordinator.acquire_file_lock(file_path):
                    try:
                        # Simulate processing time
                        time.sleep(0.1)
                        coordinator.mark_file_completed(file_path)
                        return f"{instance_id} processed {file_path.name}"
                    finally:
                        coordinator.release_file_lock(file_path)

            return f"{instance_id} skipped {file_path.name}"

        # Process files with multiple instances
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            for i, file_path in enumerate(test_files):
                instance_id = f"instance-{i % 3}"
                future = executor.submit(process_file, instance_id, file_path)
                futures.append(future)

            for future in as_completed(futures):
                results.append(future.result())

        # All files should be processed
        processed_files = [r for r in results if "processed" in r]
        assert len(processed_files) == len(test_files)

    def test_concurrent_processing_same_file(self, temp_coordination_dir, test_files):
        """Test multiple instances trying to process the same file."""

        def try_process_file(instance_id, file_path):
            coordinator = DocumentCoordinator(temp_coordination_dir, instance_id)

            # First check if we can process the file
            if not coordinator.can_process_file(file_path):
                return f"{instance_id} skipped {file_path.name} (cannot process)"

            # Try to acquire the lock
            if coordinator.acquire_file_lock(file_path):
                try:
                    # Double check if file was completed by another instance
                    if coordinator.is_file_completed(file_path):
                        return f"{instance_id} skipped {file_path.name} (completed after lock)"

                    # Simulate processing time
                    time.sleep(0.1)
                    coordinator.mark_file_completed(file_path)
                    return f"{instance_id} processed {file_path.name}"
                finally:
                    coordinator.release_file_lock(file_path)

            return f"{instance_id} skipped {file_path.name} (could not acquire lock)"

        # Multiple instances try to process the same file
        test_file = test_files[0]
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            for i in range(3):
                instance_id = f"instance-{i}"
                future = executor.submit(try_process_file, instance_id, test_file)
                futures.append(future)

            for future in as_completed(futures):
                results.append(future.result())

        # Only one instance should process the file
        processed_count = len([r for r in results if "processed" in r])
        skipped_count = len([r for r in results if "skipped" in r])

        assert processed_count == 1
        assert skipped_count == 2

    def test_file_distribution_across_instances(self, temp_coordination_dir, test_files):
        """Test that files are distributed fairly across instances."""

        def process_available_files(instance_id, all_files):
            coordinator = DocumentCoordinator(temp_coordination_dir, instance_id)
            processed_files = []

            for file_path in all_files:
                try:
                    # First check if we can process this file
                    if not coordinator.can_process_file(file_path):
                        continue

                    # Try to acquire the lock
                    if coordinator.acquire_file_lock(file_path):
                        try:
                            # Double check after acquiring lock
                            if coordinator.is_file_completed(file_path):
                                continue

                            # Simulate processing
                            time.sleep(0.05)
                            coordinator.mark_file_completed(file_path)
                            processed_files.append(file_path.name)
                        finally:
                            coordinator.release_file_lock(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

            return processed_files

        # Multiple instances process all available files
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            for i in range(3):
                instance_id = f"instance-{i}"
                future = executor.submit(process_available_files, instance_id, test_files)
                futures.append((instance_id, future))

            for instance_id, future in futures:
                results[instance_id] = future.result()

        # Check that all files were processed exactly once
        all_processed = []
        for processed_files in results.values():
            all_processed.extend(processed_files)

        assert len(all_processed) == len(test_files)
        assert len(set(all_processed)) == len(test_files)  # No duplicates


if __name__ == "__main__":
    pytest.main([__file__])

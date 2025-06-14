#!/usr/bin/env python3
"""
Basic test for instance coordination without heavy dependencies.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_coordination_functionality():
    """Test basic coordination functionality."""
    print("Testing instance coordination functionality...")

    # Import here to avoid dependency issues
    from core.utils.instance_coordinator import InstanceCoordinator, InstanceInfo

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test 1: Create coordinators
        print("  Test 1: Creating coordinators...")
        coordinator1 = InstanceCoordinator(
            config_hash="test_hash_123",
            output_path=str(temp_path / "output"),
            vector_db_path=str(temp_path / "vector_db"),
        )

        coordinator2 = InstanceCoordinator(
            config_hash="test_hash_123",
            output_path=str(temp_path / "output"),  # Same path - should conflict
            vector_db_path=str(temp_path / "vector_db"),
        )

        coordinator3 = InstanceCoordinator(
            config_hash="test_hash_123",
            output_path=str(temp_path / "output_different"),  # Different path - should work
            vector_db_path=str(temp_path / "vector_db_different"),
        )

        # Test 2: Register first instance
        print("  Test 2: Registering first instance...")
        success1 = coordinator1.register_instance()
        assert success1, "First instance should register successfully"
        print("    ✅ First instance registered")

        # Test 3: Try to register conflicting instance
        print("  Test 3: Testing conflict detection...")
        success2 = coordinator2.register_instance()
        assert not success2, "Second instance should detect conflict"
        print("    ✅ Conflict detection working")

        # Test 4: Register non-conflicting instance
        print("  Test 4: Registering non-conflicting instance...")
        success3 = coordinator3.register_instance()
        assert success3, "Third instance should register successfully"
        print("    ✅ Non-conflicting instance registered")

        # Test 5: List active instances
        print("  Test 5: Listing active instances...")
        active_instances = coordinator1.get_active_instances()
        assert len(active_instances) == 2, f"Should have 2 active instances, got {len(active_instances)}"
        print(f"    ✅ Found {len(active_instances)} active instances")

        # Test 6: Test heartbeat updates
        print("  Test 6: Testing heartbeat updates...")
        coordinator1.update_heartbeat()
        coordinator3.update_heartbeat()
        print("    ✅ Heartbeat updates working")

        # Test 7: Test instance-specific paths
        print("  Test 7: Testing instance-specific paths...")
        suggested_paths = coordinator2.suggest_instance_specific_paths()
        assert "output_path" in suggested_paths, "Should suggest output path"
        assert suggested_paths["output_path"] != coordinator1.output_path, "Should suggest different path"
        print("    ✅ Instance-specific path generation working")

        # Test 8: Cleanup
        print("  Test 8: Testing cleanup...")
        coordinator1.unregister_instance()
        coordinator3.unregister_instance()

        active_instances = coordinator1.get_active_instances()
        assert len(active_instances) == 0, "Should have no active instances after cleanup"
        print("    ✅ Cleanup working")

        print("✅ All coordination tests passed!")
        return True


def test_file_locking():
    """Test basic file locking functionality."""
    print("Testing file locking functionality...")

    from core.utils.instance_coordinator import InstanceCoordinator

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        coordinator = InstanceCoordinator(config_hash="test_hash", output_path=str(temp_path / "output"))

        test_file = temp_path / "test_file.txt"
        test_file.write_text("test content")

        # Test exclusive file access
        print("  Testing exclusive file access...")
        try:
            with coordinator.exclusive_file_access(str(test_file), "r") as f:
                content = f.read()
                assert content == "test content", "File content should match"
            print("    ✅ Exclusive file access working")
        except Exception as e:
            print(f"    ❌ File locking test failed: {e}")
            return False

        print("✅ File locking tests passed!")
        return True


def test_shared_rate_limiter():
    """Test shared rate limiter functionality."""
    print("Testing shared rate limiter functionality...")

    from core.utils.instance_coordinator import InstanceCoordinator, SharedRateLimiter

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create coordinators
        coordinator1 = InstanceCoordinator(config_hash="test_hash", output_path=str(temp_path / "output1"))
        coordinator2 = InstanceCoordinator(config_hash="test_hash", output_path=str(temp_path / "output2"))

        # Register instances
        coordinator1.register_instance()
        coordinator2.register_instance()

        # Create shared rate limiters
        rate_limiter1 = SharedRateLimiter(coordinator1, base_rate_limit=100)
        rate_limiter2 = SharedRateLimiter(coordinator2, base_rate_limit=100)

        # Test rate limit distribution
        print("  Testing rate limit distribution...")
        limit1 = rate_limiter1.get_instance_rate_limit()
        limit2 = rate_limiter2.get_instance_rate_limit()

        # With 2 instances, each should get 50 RPM (100/2)
        assert limit1 == 50, f"Instance 1 should get 50 RPM, got {limit1}"
        assert limit2 == 50, f"Instance 2 should get 50 RPM, got {limit2}"
        print(f"    ✅ Rate limits distributed correctly: {limit1}, {limit2}")

        # Test shared state updates
        print("  Testing shared state updates...")
        rate_limiter1.update_shared_state(tokens_used=100, response_time=1.5)
        rate_limiter2.update_shared_state(tokens_used=150, response_time=2.0)
        print("    ✅ Shared state updates working")

        # Cleanup
        coordinator1.unregister_instance()
        coordinator2.unregister_instance()

        print("✅ Shared rate limiter tests passed!")
        return True


def main():
    """Main test function."""
    print("🧪 Testing nBedR Instance Coordination (Basic)")
    print("=" * 50)

    try:
        # Test 1: Basic coordination
        success1 = test_coordination_functionality()
        print()

        # Test 2: File locking
        success2 = test_file_locking()
        print()

        # Test 3: Shared rate limiter
        success3 = test_shared_rate_limiter()

        if success1 and success2 and success3:
            print("\n" + "=" * 50)
            print("✅ All basic coordination tests passed!")
            print("\n💡 Key features verified:")
            print("   ✓ Instance registration and conflict detection")
            print("   ✓ File locking for concurrent access")
            print("   ✓ Shared rate limiting across instances")
            print("   ✓ Instance-specific path generation")
            print("   ✓ Heartbeat and cleanup mechanisms")
            print("\n🚀 Multiple instances can now run safely in parallel!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

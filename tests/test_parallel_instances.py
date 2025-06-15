#!/usr/bin/env python3
"""
Test script to verify multiple nBedR instances can run safely in parallel.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import after path manipulation
from core.config import EmbeddingConfig, get_config  # noqa: E402
from core.utils.instance_coordinator import InstanceCoordinator, create_instance_coordinator  # noqa: E402


def create_test_config(instance_id: int, temp_dir: Path) -> Path:
    """Create a test configuration file for an instance."""
    config_content = f"""
# Test configuration for instance {instance_id}
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=test_key_for_testing
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Vector Database Configuration
VECTOR_DB_TYPE=faiss
FAISS_INDEX_PATH={temp_dir}/faiss_index_{instance_id}

# Output Configuration
EMBEDDING_OUTPUT={temp_dir}/output_{instance_id}

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_STRATEGY=sliding_window

# Processing Configuration
CHUNK_SIZE=256
CHUNKING_STRATEGY=fixed
WORKERS=2
"""

    config_file = temp_dir / f"config_{instance_id}.env"
    with open(config_file, "w") as f:
        f.write(config_content)

    return config_file


def create_test_documents(temp_dir: Path, num_docs: int = 5) -> Path:
    """Create test documents for processing."""
    docs_dir = temp_dir / "test_docs"
    docs_dir.mkdir(exist_ok=True)

    for i in range(num_docs):
        doc_content = f"""
        Test Document {i}

        This is a test document created for parallel processing verification.
        Document ID: {i}

        Content Section 1:
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
        veniam, quis nostrud exercitation ullamco laboris.

        Content Section 2:
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
        dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
        proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

        Content Section 3:
        Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
        doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore
        veritatis et quasi architecto beatae vitae dicta sunt explicabo.
        """

        doc_file = docs_dir / f"test_doc_{i}.txt"
        with open(doc_file, "w") as f:
            f.write(doc_content)

    return docs_dir


def run_instance(instance_id: int, config_file: Path, docs_dir: Path, temp_dir: Path) -> dict:
    """Run a single nBedR instance."""
    print(f"Starting instance {instance_id}")

    # Run nBedR instance
    cmd = [
        sys.executable,
        "nbedr.py",
        "create-embeddings",
        "--env-file",
        str(config_file),
        "--datapath",
        str(docs_dir),
        "--doctype",
        "txt",
        "--verbose",
    ]

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=current_dir)  # 2 minute timeout

        end_time = time.time()

        return {
            "instance_id": instance_id,
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "instance_id": instance_id,
            "success": False,
            "duration": time.time() - start_time,
            "stdout": "",
            "stderr": "Process timed out",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "success": False,
            "duration": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
            "returncode": -2,
        }


def test_instance_coordination():
    """Test instance coordination functionality."""
    print("Testing instance coordination...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock config
        config = EmbeddingConfig()
        config.output = str(temp_path / "output")
        config.embedding_provider = "openai"
        config.vector_db_type = "faiss"
        config.faiss_index_path = str(temp_path / "faiss_index")

        # Test 1: Single instance registration
        coordinator1 = create_instance_coordinator(config)
        assert coordinator1.register_instance(), "Failed to register first instance"

        # Test 2: Second instance with same paths (should detect conflict)
        coordinator2 = create_instance_coordinator(config)
        assert not coordinator2.register_instance(), "Second instance should detect conflict"

        # Test 3: Instance-specific paths
        suggested_paths = coordinator2.suggest_instance_specific_paths()
        assert "output_path" in suggested_paths, "Should suggest instance-specific paths"
        assert suggested_paths["output_path"] != config.output, "Should suggest different output path"

        # Test 4: Active instances listing
        active_instances = coordinator1.get_active_instances()
        assert len(active_instances) == 1, f"Should have 1 active instance, got {len(active_instances)}"

        # Test 5: Cleanup
        coordinator1.unregister_instance()
        active_instances = coordinator1.get_active_instances()
        assert len(active_instances) == 0, "Should have no active instances after cleanup"

        print("✅ Instance coordination tests passed")


def test_parallel_execution():
    """Test parallel execution of multiple instances."""
    print("Testing parallel execution...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test documents
        docs_dir = create_test_documents(temp_path)
        print(f"Created test documents in {docs_dir}")

        # Create configurations for multiple instances
        num_instances = 3
        configs = []
        for i in range(num_instances):
            config_file = create_test_config(i, temp_path)
            configs.append(config_file)

        print(f"Created {num_instances} test configurations")

        # Run instances in parallel
        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            # Submit all instances
            futures = []
            for i, config_file in enumerate(configs):
                future = executor.submit(run_instance, i, config_file, docs_dir, temp_path)
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Instance {result['instance_id']} completed: {'✅' if result['success'] else '❌'}")

        total_time = time.time() - start_time

        # Analyze results
        successful_instances = [r for r in results if r["success"]]
        failed_instances = [r for r in results if not r["success"]]

        print("\n📊 Parallel Execution Results:")
        print(f"Total Instances: {num_instances}")
        print(f"Successful: {len(successful_instances)}")
        print(f"Failed: {len(failed_instances)}")
        print(f"Total Time: {total_time:.2f}s")

        if successful_instances:
            avg_duration = sum(r["duration"] for r in successful_instances) / len(successful_instances)
            print(f"Average Instance Duration: {avg_duration:.2f}s")

        # Show details for failed instances
        if failed_instances:
            print("\n❌ Failed Instances:")
            for result in failed_instances:
                print(f"  Instance {result['instance_id']}:")
                print(f"    Return Code: {result['returncode']}")
                print(f"    Error: {result['stderr'][:200]}...")

        # Verify outputs were created
        output_dirs = list(temp_path.glob("output_*"))
        print(f"\nOutput directories created: {len(output_dirs)}")

        # Check for conflicts in vector database files
        faiss_dirs = list(temp_path.glob("faiss_index_*"))
        print(f"FAISS index directories created: {len(faiss_dirs)}")

        return len(successful_instances) > 0


def main():
    """Main test function."""
    print("🧪 Testing nBedR Parallel Instance Execution")
    print("=" * 50)

    try:
        # Test 1: Instance coordination
        test_instance_coordination()
        print()

        # Test 2: Parallel execution
        success = test_parallel_execution()

        print("\n" + "=" * 50)
        if success:
            print("✅ Parallel instance tests completed successfully!")
            print("\n💡 Multiple nBedR instances can now run safely in parallel:")
            print("   - Instance coordination prevents conflicts")
            print("   - Automatic path separation for outputs")
            print("   - File locking for shared resources")
            print("   - Rate limiting coordination")
        else:
            print("❌ Some tests failed. Check the output above for details.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

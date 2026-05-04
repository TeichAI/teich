#!/usr/bin/env python3
"""Manual integration test script for teich.

Run this to verify everything works end-to-end:
    python scripts/integration_test.py

Requirements:
    - Docker running
    - OPENAI_API_KEY env var set
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_docker():
    """Verify Docker is available."""
    print("Checking Docker...")
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("❌ Docker not available. Is Docker Desktop running?")
        return False
    print("✅ Docker available")
    return True


def check_api_key():
    """Verify API key is set."""
    print("Checking API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("   Run: export OPENAI_API_KEY=sk-...")
        return False
    print("✅ API key found")
    return True


def build_docker_image():
    """Build the Docker runtime image."""
    print("\nBuilding Docker image (this may take 2-3 minutes)...")
    dockerfile = Path(__file__).parent.parent / "docker" / "codex-runtime.Dockerfile"

    result = subprocess.run(
        [
            "docker", "build",
            "-f", str(dockerfile),
            "-t", "teich-runtime:v3",
            str(dockerfile.parent),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("❌ Docker build failed")
        print(result.stderr)
        return False

    print("✅ Docker image built")
    return True


def test_codex_in_container():
    """Test that codex runs inside container."""
    print("\nTesting Codex CLI in container...")

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-e", f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}",
            "teich-runtime:v3",
            "codex", "--version",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        print("❌ Codex CLI not working in container")
        print(result.stderr)
        return False

    print(f"✅ Codex CLI version: {result.stdout.strip()}")
    return True


def test_python_in_container():
    """Test that Python is available inside container."""
    print("\nTesting Python in container...")

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "teich-runtime:v3",
            "bash", "-lc", "python --version && python3 --version",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        print("❌ Python not working in container")
        print(result.stderr)
        return False

    print(f"✅ Python available: {result.stdout.strip()}")
    return True


def test_init_command():
    """Test the init CLI command."""
    print("\nTesting 'init' command...")

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-m", "teich.cli", "init", tmp],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode != 0:
            print("❌ Init command failed")
            print(result.stderr)
            return False

        config_file = Path(tmp) / "config.yaml"
        prompts_file = Path(tmp) / "prompts.txt"

        if not config_file.exists():
            print("❌ config.yaml not created")
            return False
        if not prompts_file.exists():
            print("❌ prompts.txt not created")
            return False

        print("✅ Init command works")
        return True


def test_generate_with_real_api():
    """Test actual generation with OpenAI API."""
    print("\nTesting full generation with OpenAI API...")
    print("This will cost ~$0.01-0.05 for a simple prompt")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create minimal config
        config_content = f"""
model:
  model: codex-mini-latest
  approval_mode: none

prompts:
  - Write a Python function to calculate fibonacci numbers

output:
  traces_dir: {tmp_path}/output

openai_api_key: {os.getenv('OPENAI_API_KEY')}
timeout_seconds: 120
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        print(f"   Config: {config_file}")
        print(f"   Output: {tmp_path}/output")
        print("   Running generation (timeout 2 min)...")

        result = subprocess.run(
            [
                sys.executable, "-m", "teich.cli",
                "generate", "-c", str(config_file),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=150,
        )

        if result.returncode != 0:
            print("❌ Generation failed")
            print(result.stderr)
            return False

        # Check output
        output_dir = tmp_path / "output"
        if not output_dir.exists():
            print("❌ Output directory not created")
            return False

        trace_files = list(output_dir.glob("*.jsonl"))
        if not trace_files:
            print("❌ No trace files generated")
            return False

        print(f"✅ Generated {len(trace_files)} trace file(s)")
        print(f"   First file: {trace_files[0].name}")

        # Validate trace format
        with open(trace_files[0]) as f:
            first_line = f.readline()
            import json
            try:
                event = json.loads(first_line)
                if event.get("type") == "session":
                    print("✅ Valid trace format (session event first)")
                else:
                    print(f"⚠️ Unexpected first event type: {event.get('type')}")
            except json.JSONDecodeError:
                print("❌ Invalid JSON in trace file")
                return False

        return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Teich - Integration Test Suite")
    print("=" * 60)

    # Pre-flight checks
    if not check_docker():
        sys.exit(1)
    if not check_api_key():
        sys.exit(1)

    # Tests
    tests = [
        ("Docker Build", build_docker_image),
        ("Codex in Container", test_codex_in_container),
        ("Init Command", test_init_command),
        ("Full Generation", test_generate_with_real_api),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)

    if all_passed:
        print("\n🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

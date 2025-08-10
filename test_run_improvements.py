#!/usr/bin/env python3
"""
Test script to demonstrate the improved run.py functionality
"""

import subprocess
import sys
import os

def test_help_output():
    """Test the help output functionality"""
    print("=== Testing Help Output ===")
    result = subprocess.run([sys.executable, "run.py", "--help"], capture_output=True, text=True)
    print("Return code:", result.returncode)
    print("Help output:")
    print(result.stdout)
    return result.returncode == 0

def test_video_validation():
    """Test video validation functionality"""
    print("\n=== Testing Video Validation ===")
    
    # Test with existing video
    result = subprocess.run([
        sys.executable, "run.py", 
        "--video", "examples/videos/rally.mp4",
        "--verbose"
    ], capture_output=True, text=True, timeout=30)
    
    print("Return code:", result.returncode)
    print("Output (first 1000 chars):")
    print(result.stdout[:1000])
    if result.stderr:
        print("Errors (first 500 chars):")
        print(result.stderr[:500])
    
    # Check if video validation occurred
    return "Video validated" in result.stdout

def test_invalid_video():
    """Test handling of invalid video path"""
    print("\n=== Testing Invalid Video Handling ===")
    
    result = subprocess.run([
        sys.executable, "run.py", 
        "--video", "nonexistent_video.mp4",
        "--auto-clean",
        "--verbose"
    ], capture_output=True, text=True, timeout=10)
    
    print("Return code:", result.returncode)
    print("Output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    # Should fail gracefully
    return result.returncode != 0 and "invalid" in result.stdout.lower()

def main():
    """Run all tests"""
    print("Testing improved run.py script functionality...")
    
    tests = [
        ("Help Output", test_help_output),
        ("Video Validation", test_video_validation), 
        ("Invalid Video Handling", test_invalid_video)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✓ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: ERROR - {e}")
    
    print(f"\n=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple test script to verify that the demonstrator __init__.py works correctly.
"""

if __name__ == "__main__":
    # Import directly from the module
    from dtaianomaly.demonstrator import run_demonstrator
    
    # Run the demonstrator
    print("Starting the demonstrator...")
    run_demonstrator() 
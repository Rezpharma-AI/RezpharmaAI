#!/usr/bin/env python3
"""
Healthcheck script for Railway deployment.
Verifies that all required modules can be imported before starting the API.
"""

import sys
import os

print("[STARTUP] Checking dependencies...")

try:
    import fastapi
    print("✓ FastAPI available")
    import uvicorn
    print("✓ Uvicorn available")
    import numpy
    print("✓ NumPy available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Check if database will be available
DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "rezpharma.db")
DB_URL = os.getenv("DB_DOWNLOAD_URL")

if DB_URL:
    print(f"[STARTUP] Database bootstrap configured (DB_DOWNLOAD_URL set)")
else:
    print("[WARNING] DB_DOWNLOAD_URL not set - ensure database exists locally")

print("[STARTUP] All checks passed! Ready to start API server.")
sys.exit(0)

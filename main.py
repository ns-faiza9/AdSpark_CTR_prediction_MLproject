"""Main entry point for AdSpark CTR Prediction Flask Application.

Run this script to launch the Flask development web server:
    python main.py
"""
import sys
import os
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print("  AdSpark — Click-Through Rate (CTR) Prediction Flask App")
    print(f"  Server running at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=True)

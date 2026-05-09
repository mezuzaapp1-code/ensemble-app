"""
CLI: ZIP the project into backups/stable_v{timestamp}.zip (same rules as the dashboard backup).
Run: python backup_working.py
"""
from pathlib import Path

from founder_status import create_stable_backup_zip

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    out = create_stable_backup_zip(base)
    print("Created:", out)

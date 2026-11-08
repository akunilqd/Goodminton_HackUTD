"""Clear the current application's six SpacetimeDB tables.

Local outputs/jobs are deliberately cleaned separately after checking active workers
and resolving their absolute paths. Usage: python tools/clear_all.py --yes
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stdb_client import StdbClient

TABLES = ("run_meta", "running_details", "shot_class", "summary", "rally", "analyzed_video")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Confirm deleting all analysis rows")
    args = parser.parse_args()
    if not args.yes:
        parser.error("Pass --yes to delete all current analysis data.")
    client = StdbClient()
    print(f"Target: {client.server}/{client.module}")
    # Include orphan rows whose metadata was never published or already removed.
    run_ids = set()
    for table in TABLES:
        run_ids.update(r["run_id"] for r in client._sql(f"SELECT run_id FROM {table}"))
    print(f"Clearing {len(run_ids)} run IDs")
    for run_id in sorted(run_ids):
        client.clear_run(run_id)  # Fail immediately on rejected/failed reducer calls.
    remaining = {table: int(client._sql(f"SELECT COUNT(*) AS count FROM {table}")[0]["count"])
                 for table in TABLES}
    print(remaining)
    if any(remaining.values()):
        raise RuntimeError("Database is not empty; another publisher may still be running.")
    print("Verified: all six tables are empty.")

if __name__ == "__main__":
    main()

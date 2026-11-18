#!/usr/bin/env python3
"""Export local SQLite request_logs + eval_runs to fixtures/db_seed.json for Railway import."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "orchestrator.sqlite"
OUT = ROOT / "fixtures" / "db_seed.json"


def _ser(v):
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"Missing {DB}")
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    eval_runs = [dict(r) for r in cur.execute("SELECT * FROM eval_runs ORDER BY created_at")]
    logs = [dict(r) for r in cur.execute("SELECT * FROM request_logs ORDER BY id")]

    payload = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "source": str(DB),
        "eval_runs": [{k: _ser(v) for k, v in row.items()} for row in eval_runs],
        "request_logs": [{k: _ser(v) for k, v in row.items()} for row in logs],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT} ({len(eval_runs)} eval_runs, {len(logs)} request_logs)")


if __name__ == "__main__":
    main()

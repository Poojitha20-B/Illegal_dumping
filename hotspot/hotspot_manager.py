import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "hotspot.db"

def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hotspots (
                hotspot_id     TEXT PRIMARY KEY,
                location_name  TEXT,
                violation_count INTEGER DEFAULT 0,
                status         TEXT DEFAULT 'active',
                first_seen     TEXT,
                last_seen      TEXT,
                latitude       REAL,
                longitude      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violation_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                challan_id TEXT,
                location  TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hotspot_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                hotspot_id TEXT,
                event_id   INTEGER
            )
        """)
        conn.commit()

class HotspotManager:
    def __init__(self):
        _init_db()

    def get_all_hotspots(self):
        with _get_conn() as conn:
            rows = conn.execute("SELECT * FROM hotspots ORDER BY violation_count DESC").fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self):
        with _get_conn() as conn:
            rows = conn.execute("SELECT * FROM violation_events ORDER BY timestamp DESC LIMIT 200").fetchall()
        return [dict(r) for r in rows]

    def run_hotspot_check(self):
        return []

    def summary(self):
        with _get_conn() as conn:
            total  = conn.execute("SELECT COUNT(*) FROM hotspots").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM hotspots WHERE status='active'").fetchone()[0]
        return {"total_hotspots": total, "active_hotspots": active}
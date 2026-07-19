# migrate_db.py
import sqlite3

DB_PATH = "database/attendance.db"

expected_columns = {
    "students": [
        ("student_name", "TEXT PRIMARY KEY"),
        ("email", "TEXT"),
        ("phone", "TEXT"),
        ("embedding", "BLOB"),
        ("enrolled_at", "TEXT")
    ],
    "sessions": [
        ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("session_id", "TEXT"),
        ("date", "TEXT"),
        ("session_name", "TEXT DEFAULT ''"),
        ("duration_minutes", "INTEGER DEFAULT 30"),
        ("status", "TEXT DEFAULT 'OPEN'")
    ],
    "attendance": [
        ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("student_name", "TEXT"),
        ("session_id", "TEXT"),
        ("status", "TEXT DEFAULT 'PRESENT'"),
        ("timestamp", "TEXT")
    ]
}

def get_existing_columns(con, table):
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return [r[1] for r in rows]

def add_column(con, table, col_def):
    cur = con.cursor()
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
        con.commit()
        print(f"✅ Added column to {table}: {col_def}")
    except Exception as e:
        print(f"⚠️ Failed to add column {col_def} to {table}: {e}")

def ensure_table(con, table, cols):
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    exists = cur.fetchone()
    if not exists:
        print(f"➕ Table {table} does not exist — creating.")
        cols_sql = ", ".join([f"{name} {typ}" for name, typ in cols])
        cur.execute(f"CREATE TABLE {table} ({cols_sql})")
        con.commit()
        return

    existing = get_existing_columns(con, table)
    print(f"ℹ️ Existing columns for {table}: {existing}")
    for name, typ in cols:
        if name not in existing:
            add_column(con, table, f"{name} {typ}")

def main():
    con = sqlite3.connect(DB_PATH)
    try:
        for table, cols in expected_columns.items():
            ensure_table(con, table, cols)
    finally:
        con.close()
    print("🎉 Migration finished. Restart your app now.")

if __name__ == "__main__":
    main()

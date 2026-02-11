import sqlite3
import json
import os

DB_FILE = "benchmarks.db"
JSON_FILE = "model_results.json"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_path TEXT,
        context_window INTEGER,
        source_text_path TEXT,
        timestamp TEXT,
        original_size INTEGER,
        compressed_size INTEGER,
        compression_ratio REAL,
        bpc REAL,
        execution_time REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        source_url TEXT NOT NULL,
        text_content TEXT NOT NULL,
        collected_at TEXT NOT NULL,
        char_count INTEGER NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_source_benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_source_id INTEGER,
        model_name TEXT,
        model_path TEXT,
        context_window INTEGER,
        timestamp TEXT,
        original_size INTEGER,
        compressed_size INTEGER,
        compression_ratio REAL,
        bpc REAL,
        execution_time REAL,
        FOREIGN KEY(data_source_id) REFERENCES data_sources(id)
    )
    ''')

    conn.commit()
    return conn


def _make_slug(category, title):
    """Reproduce the slug logic from populate_sources.py."""
    return f"{category}_{title.lower().replace(' ', '_').replace('/', '_')[:40]}"


def migrate_data_source_benchmarks(conn):
    """Copy matching rows from benchmarks → data_source_benchmarks."""
    cursor = conn.cursor()

    # Check if already migrated
    cursor.execute("SELECT COUNT(*) FROM data_source_benchmarks")
    existing = cursor.fetchone()[0]
    if existing > 0:
        print(
            f"data_source_benchmarks already has {existing} rows. Skipping migration.")
        return

    # Build slug → data_source_id map
    cursor.execute("SELECT id, category, title FROM data_sources")
    slug_map = {}
    for ds_id, category, title in cursor.fetchall():
        slug = _make_slug(category, title)
        path = f"source_text/{slug}.txt"
        slug_map[path] = ds_id

    # Copy matching benchmarks rows
    count = 0
    for path, ds_id in slug_map.items():
        cursor.execute('''
            SELECT model_name, model_path, context_window, timestamp,
                   original_size, compressed_size, compression_ratio, bpc, execution_time
            FROM benchmarks
            WHERE source_text_path = ?
        ''', (path,))
        for row in cursor.fetchall():
            cursor.execute('''
                INSERT INTO data_source_benchmarks (
                    data_source_id, model_name, model_path, context_window,
                    timestamp, original_size, compressed_size,
                    compression_ratio, bpc, execution_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (ds_id, *row))
            count += 1

    conn.commit()
    print(f"Migrated {count} rows into data_source_benchmarks.")


def migrate_json(conn):
    if not os.path.exists(JSON_FILE):
        print(f"{JSON_FILE} not found. Skipping migration.")
        return

    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)

        cursor = conn.cursor()
        count = 0
        for entry in data:
            # Check if entry already exists (simple check by timestamp and model_path)
            cursor.execute('''
                SELECT id FROM benchmarks 
                WHERE timestamp = ? AND model_path = ?
            ''', (entry.get('timestamp'), entry.get('model_path')))

            if cursor.fetchone() is None:
                cursor.execute('''
                    INSERT INTO benchmarks (
                        model_name, model_path, context_window, source_text_path,
                        timestamp, original_size, compressed_size, 
                        compression_ratio, bpc, execution_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.get('model_name', 'Unknown'),
                    entry.get('model_path'),
                    # Default to 50 if missing
                    entry.get('context_window', 50),
                    entry.get('source_text_path',
                              'short_story.txt'),  # Default
                    entry.get('timestamp'),
                    entry.get('original_size_bytes'),
                    entry.get('compressed_size_bytes'),
                    entry.get('compression_ratio'),
                    entry.get('bits_per_character'),
                    entry.get('execution_time_seconds')
                ))
                count += 1

        conn.commit()
        print(f"Migrated {count} entries from {JSON_FILE}.")

    except json.JSONDecodeError:
        print(f"Error decoding {JSON_FILE}.")
    except Exception as e:
        print(f"Migration error: {e}")


def main():
    print(f"Initializing {DB_FILE}...")
    conn = init_db()
    migrate_json(conn)
    migrate_data_source_benchmarks(conn)
    conn.close()
    print("Database setup complete.")


if __name__ == "__main__":
    main()

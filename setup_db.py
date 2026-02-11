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
    
    conn.commit()
    return conn

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
                    entry.get('context_window', 50), # Default to 50 if missing
                    entry.get('source_text_path', 'short_story.txt'), # Default
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
    conn.close()
    print("Database setup complete.")

if __name__ == "__main__":
    main()

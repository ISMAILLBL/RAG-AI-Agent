import sqlite3, json
conn = sqlite3.connect("rag.db")
c = conn.cursor()

# 1) Liste des tables
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
print("Tables:", [t[0] for t in tables])

# 2) Schéma complet
for (name,) in tables:
    print("\n===", name, "===")
    print(c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()[0])
    print("Columns:")
    cols = c.execute(f"PRAGMA table_info({name})").fetchall()
    for col in cols:
        # (cid, name, type, notnull, dflt_value, pk)
        print(f"  - {col[1]} {col[2]}{' NOT NULL' if col[3] else ''}{' PK' if col[5] else ''}")
    fks = c.execute(f"PRAGMA foreign_key_list({name})").fetchall()
    if fks:
        print("Foreign Keys:")
        for fk in fks:
            # (id, seq, table, from, to, on_update, on_delete, match)
            print(f"  - {fk[3]} → {fk[2]}.{fk[4]} (on_delete={fk[6]}, on_update={fk[5]})")
conn.close()

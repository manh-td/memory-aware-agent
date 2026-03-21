"""Vector index and memory management utilities."""


def safe_create_index(conn, vs, idx_name):
    """Create IVF vector index using raw SQL for maximum compatibility.

    Uses IVF (NEIGHBOR PARTITIONS) instead of HNSW to avoid:
    - ORA-00600 on some Oracle Free versions
    - ORA-51928 (DML not supported with INMEMORY NEIGHBOR GRAPH)
    - ORA-51962 (vector memory pool sizing issues)

    Handles ORA-00955 (index already exists) by skipping.
    """
    dist_map = {
        "COSINE": "COSINE",
        "EUCLIDEAN_DISTANCE": "EUCLIDEAN",
        "DOT_PRODUCT": "DOT",
    }
    dist = dist_map.get(vs.distance_strategy.name, "COSINE")

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE VECTOR INDEX {idx_name}"
                f" ON {vs.table_name}(EMBEDDING)"
                f" ORGANIZATION NEIGHBOR PARTITIONS"
                f" DISTANCE {dist}"
                f" WITH TARGET ACCURACY 95"
            )
        print(f"  ✅ Created index: {idx_name}")
    except Exception as e:
        err = str(e)
        if "ORA-00955" in err:
            print(f"  ⏭️  Index already exists: {idx_name}")
        else:
            raise


def cleanup_vector_memory(conn, drop_tables: bool = False, table_prefix: str = None):
    """
    Clean up vector indexes and optionally tables to free up vector memory space.

    Use this when you encounter ORA-51962: vector memory area is out of space.

    Args:
        conn: Oracle database connection
        drop_tables: If True, also drops the vector tables (WARNING: deletes all data)
        table_prefix: If provided, only clean up tables/indexes matching this prefix
                      (e.g., "SEMANTIC" to only clean SEMANTIC_MEMORY)

    Returns:
        dict with counts of dropped indexes and tables
    """
    dropped_indexes = 0
    dropped_tables = 0

    print("=" * 60)
    print("🧹 CLEANING UP VECTOR MEMORY")
    print("=" * 60)

    with conn.cursor() as cur:
        # Find all vector indexes
        cur.execute("""
            SELECT INDEX_NAME, TABLE_NAME 
            FROM USER_INDEXES 
            WHERE INDEX_TYPE = 'VECTOR'
            ORDER BY TABLE_NAME
        """)
        indexes = cur.fetchall()

        if not indexes:
            print("  ℹ️ No vector indexes found")
        else:
            print(f"\n[1/2] Dropping vector indexes ({len(indexes)} found)...")
            for idx_name, table_name in indexes:
                # Apply prefix filter if specified
                if table_prefix and not table_name.upper().startswith(table_prefix.upper()):
                    continue
                try:
                    cur.execute(f"DROP INDEX {idx_name}")
                    print(f"  ✅ Dropped index: {idx_name} (on {table_name})")
                    dropped_indexes += 1
                except Exception as e:
                    print(f"  ⚠️ Failed to drop {idx_name}: {e}")
            conn.commit()

        if drop_tables:
            # Find vector tables (tables with VECTOR columns)
            cur.execute("""
                SELECT DISTINCT TABLE_NAME 
                FROM USER_TAB_COLUMNS 
                WHERE DATA_TYPE = 'VECTOR'
                ORDER BY TABLE_NAME
            """)
            tables = cur.fetchall()

            if not tables:
                print("  ℹ️ No vector tables found")
            else:
                print(f"\n[2/2] Dropping vector tables ({len(tables)} found)...")
                for (table_name,) in tables:
                    # Apply prefix filter if specified
                    if table_prefix and not table_name.upper().startswith(table_prefix.upper()):
                        continue
                    try:
                        cur.execute(f"DROP TABLE {table_name} PURGE")
                        print(f"  ✅ Dropped table: {table_name}")
                        dropped_tables += 1
                    except Exception as e:
                        print(f"  ⚠️ Failed to drop {table_name}: {e}")
                conn.commit()
        else:
            print("\n[2/2] Skipping table deletion (drop_tables=False)")
            print("  💡 Set drop_tables=True to also remove tables and free more space")

    print("\n" + "=" * 60)
    print(f"🎉 CLEANUP COMPLETE: {dropped_indexes} indexes, {dropped_tables} tables dropped")
    print("=" * 60)

    return {"indexes_dropped": dropped_indexes, "tables_dropped": dropped_tables}


def list_vector_objects(conn):
    """
    List all vector indexes and tables in the current schema.
    Useful for diagnosing space issues before cleanup.
    """
    print("=" * 60)
    print("📋 VECTOR OBJECTS IN SCHEMA")
    print("=" * 60)

    with conn.cursor() as cur:
        # List vector indexes
        cur.execute("""
            SELECT INDEX_NAME, TABLE_NAME, STATUS
            FROM USER_INDEXES 
            WHERE INDEX_TYPE = 'VECTOR'
            ORDER BY TABLE_NAME
        """)
        indexes = cur.fetchall()

        print(f"\n🔍 Vector Indexes ({len(indexes)}):")
        if indexes:
            for idx_name, table_name, status in indexes:
                print(f"  - {idx_name} on {table_name} [{status}]")
        else:
            print("  (none)")

        # List tables with vector columns
        cur.execute("""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM USER_TAB_COLUMNS 
            WHERE DATA_TYPE = 'VECTOR'
            ORDER BY TABLE_NAME
        """)
        tables = cur.fetchall()

        print(f"\n📊 Tables with Vector Columns ({len(tables)}):")
        if tables:
            for table_name, col_name in tables:
                # Get row count
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    print(f"  - {table_name}.{col_name} ({count:,} rows)")
                except:
                    print(f"  - {table_name}.{col_name}")
        else:
            print("  (none)")

    print("=" * 60)

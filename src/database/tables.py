"""Table registry and creation utilities."""


def table_exists(conn, table_name):
    """Check if a table exists in the current user's schema."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) 
            FROM USER_TABLES 
            WHERE TABLE_NAME = UPPER(:table_name)
        """, {"table_name": table_name})
        return cur.fetchone()[0] > 0


def create_conversational_history_table(conn, table_name: str = "CONVERSATIONAL_MEMORY"):
    """
    Create a table to store conversational history.
    If the table already exists, returns the table name without recreating it.
    """
    # Check if table already exists
    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists (using existing table)")
        return table_name

    with conn.cursor() as cur:
        # Create table with proper schema
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                thread_id VARCHAR2(100) NOT NULL,
                role VARCHAR2(50) NOT NULL,
                content CLOB NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata CLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary_id VARCHAR2(100) DEFAULT NULL
            )
        """)

        # Create index on thread_id for faster lookups
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)
        """)

        # Create index on timestamp for ordering
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)
        """)

    conn.commit()
    print(f"  ✅ Table {table_name} created successfully with indexes")
    return table_name


def create_tool_log_table(conn, table_name: str = "TOOL_LOG_MEMORY"):
    """
    Create a table to store raw tool execution logs per thread.
    If the table already exists, returns the table name without recreating it.
    """
    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists (using existing table)")
        return table_name

    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                thread_id VARCHAR2(100) NOT NULL,
                tool_call_id VARCHAR2(200),
                tool_name VARCHAR2(200) NOT NULL,
                tool_args CLOB,
                result CLOB,
                result_preview VARCHAR2(2000),
                status VARCHAR2(30) DEFAULT 'success',
                error_message CLOB,
                metadata CLOB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)
        """)
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_tool_name ON {table_name}(tool_name)
        """)
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)
        """)

    conn.commit()
    print(f"  ✅ Table {table_name} created successfully with indexes")
    return table_name

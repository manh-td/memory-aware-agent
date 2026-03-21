import os

from src.utils.helpers import suppress_warnings, load_env
from src.database.connection import setup_oracle_database, connect_to_oracle
from src.config import (
    ORACLE_VECTOR_USER,
    ORACLE_VECTOR_PASSWORD,
    ORACLE_DEFAULT_DSN,
    ORACLE_PROGRAM_NAME,
    ALL_TABLES
)


suppress_warnings()
load_env()
setup_oracle_database()


# Connect as the VECTOR user for all subsequent operations
database_connection = connect_to_oracle(
    user=ORACLE_VECTOR_USER,
    password=ORACLE_VECTOR_PASSWORD,
    dsn=os.getenv("ORACLE_DSN", ORACLE_DEFAULT_DSN),
    program=ORACLE_PROGRAM_NAME,
)


print("Using user:", database_connection.username)


# Drop existing tables to start fresh
for table in ALL_TABLES:
    try:
        with database_connection.cursor() as cur:
            cur.execute(f"DROP TABLE {table} PURGE")
            print(f"  - {table} (dropped)")
    except Exception as e:
        if "ORA-00942" in str(e):
            print(f"  - {table} (not exists)")
        else:
            print(f"  ✗ {table}: {e}")
            
            
database_connection.commit()
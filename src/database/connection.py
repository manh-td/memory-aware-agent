"""Oracle database setup and connection utilities."""

import os
import time
import oracledb

from src.config import (
    ORACLE_DEFAULT_ADMIN_USER,
    ORACLE_DEFAULT_ADMIN_PASSWORD,
    ORACLE_DEFAULT_DSN,
    ORACLE_VECTOR_USER,
    ORACLE_VECTOR_PASSWORD,
    ORACLE_CONNECTION_RETRIES,
    ORACLE_CONNECTION_DELAY_SECONDS,
)


def setup_oracle_database(
    admin_user=ORACLE_DEFAULT_ADMIN_USER,
    admin_password=ORACLE_DEFAULT_ADMIN_PASSWORD,
    dsn=ORACLE_DEFAULT_DSN,
    vector_password=ORACLE_VECTOR_PASSWORD,
):
    """
    One-time admin setup: configures tablespace and VECTOR user.

    Requires an admin user (e.g. system). This function:
    1. Connects as admin
    2. Finds an ASSM tablespace via USER_TABLESPACES (fix ORA-43853)
    3. Creates VECTOR user with required grants and ASSM default tablespace
    4. Tests connection as VECTOR
    """
    print("=" * 60)
    print("ORACLE DATABASE SETUP")
    print("=" * 60)

    # Step 1: Connect as admin
    print("\n[1/4] Connecting as admin...")
    try:
        admin_conn = oracledb.connect(
            user=admin_user, password=admin_password, dsn=dsn
        )
        print(f"  Connected as {admin_user}")
    except Exception as e:
        print(f"  Admin connection failed: {e}")
        return False

    try:
        # Step 2: Find ASSM tablespace for JSON column support
        print("\n[2/4] Finding JSON-compatible (ASSM) tablespace...")
        assm_ts = _find_assm_tablespace(admin_conn)

        # Step 3: Create VECTOR user with ASSM default tablespace
        print("\n[3/4] Creating VECTOR user...")
        with admin_conn.cursor() as cur:
            ts_clause = (
                f"DEFAULT TABLESPACE {assm_ts}" if assm_ts else ""
            )
            cur.execute(f"""
                DECLARE
                    user_count NUMBER;
                BEGIN
                    SELECT COUNT(*) INTO user_count
                    FROM all_users WHERE username = 'VECTOR';
                    IF user_count = 0 THEN
                        EXECUTE IMMEDIATE
                            'CREATE USER VECTOR IDENTIFIED BY '
                            || '{vector_password} {ts_clause}';
                        EXECUTE IMMEDIATE
                            'GRANT CONNECT, RESOURCE, CREATE SESSION'
                            || ' TO VECTOR';
                        EXECUTE IMMEDIATE
                            'GRANT UNLIMITED TABLESPACE TO VECTOR';
                        EXECUTE IMMEDIATE
                            'GRANT CREATE TABLE, CREATE SEQUENCE,'
                            || ' CREATE VIEW TO VECTOR';
                    END IF;
                END;
            """)
            # Always set the default tablespace for VECTOR (even
            # if the user already existed from a previous run)
            if assm_ts:
                cur.execute(
                    f"ALTER USER VECTOR DEFAULT TABLESPACE"
                    f" {assm_ts}"
                )
        admin_conn.commit()
        if assm_ts:
            print(f"  VECTOR user ready "
                  f"(default tablespace: {assm_ts})")
        else:
            print("  VECTOR user created but no ASSM tablespace"
                  " found — JSON columns may fail (ORA-43853)")

    except Exception as e:
        print(f"  Warning during setup: {e}")
    finally:
        admin_conn.close()

    # Step 4: Test connection as VECTOR
    print("\n[4/4] Testing connection as VECTOR...")
    try:
        conn = oracledb.connect(
            user=ORACLE_VECTOR_USER, password=vector_password, dsn=dsn
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM dual")
            cur.fetchone()
        conn.close()
        print("  Connection successful!")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"""
You can now connect to Oracle:
    User: VECTOR
    Password: {vector_password}
    DSN: {dsn}
""")
    return True


def _find_assm_tablespace(conn):
    """
    Find an existing ASSM tablespace for JSON column support.

    Uses USER_TABLESPACES which is accessible to ANY Oracle user
    (no DBA privileges required). Prefers DATA > USERS > SYSAUX.
    Only attempts to CREATE a tablespace as a last resort.

    Returns the tablespace name or None.
    """
    # Step 1: Query USER_TABLESPACES for existing ASSM tablespaces
    # This view is available to every Oracle user — no DBA needed.
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT TABLESPACE_NAME
                FROM USER_TABLESPACES
                WHERE SEGMENT_SPACE_MANAGEMENT = 'AUTO'
                  AND STATUS = 'ONLINE'
                ORDER BY CASE TABLESPACE_NAME
                    WHEN 'DATA' THEN 1
                    WHEN 'USERS' THEN 2
                    WHEN 'SYSAUX' THEN 3
                    ELSE 4
                END
            """)
            row = cur.fetchone()
            if row:
                print(f"  Found ASSM tablespace: {row[0]}")
                return row[0]
    except Exception as e:
        print(f"  USER_TABLESPACES query failed: {e}")

    # Step 2: No ASSM tablespace found — try creating DATA
    # Try with OMF first, then with explicit path if possible
    create_sqls = [
        "CREATE TABLESPACE DATA"
        " DATAFILE SIZE 500M"
        " AUTOEXTEND ON NEXT 100M MAXSIZE UNLIMITED"
        " SEGMENT SPACE MANAGEMENT AUTO"
    ]
    # Try to discover datafile path for non-OMF installs
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT FILE_NAME FROM DBA_DATA_FILES"
                " FETCH FIRST 1 ROW ONLY"
            )
            row = cur.fetchone()
            if row:
                datafile_dir = os.path.dirname(row[0])
                create_sqls.insert(0,
                                   f"CREATE TABLESPACE DATA"
                                   f" DATAFILE '{datafile_dir}/data01.dbf'"
                                   f" SIZE 500M AUTOEXTEND ON NEXT 100M"
                                   f" MAXSIZE UNLIMITED"
                                   f" SEGMENT SPACE MANAGEMENT AUTO"
                                   )
    except Exception:
        pass

    for sql in create_sqls:
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
            print("  Created DATA tablespace (ASSM)")
            return 'DATA'
        except Exception as e:
            err = str(e)
            if "ORA-01543" in err:
                print("  DATA tablespace already exists")
                return 'DATA'
            continue

    print("  Could not find or create ASSM tablespace")
    return None


def connect_to_oracle(
    max_retries=ORACLE_CONNECTION_RETRIES,
    retry_delay=ORACLE_CONNECTION_DELAY_SECONDS,
    user=ORACLE_VECTOR_USER,
    password=ORACLE_VECTOR_PASSWORD,
    dsn=ORACLE_DEFAULT_DSN,
    program="langchain_oracledb_deep_research_demo",
):
    """
    Connect to Oracle database with retry logic and better error handling.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries
        user: Database user to connect as
        password: Database user password
        dsn: Database DSN/connection string
        program: Program identifier for the connection
    """

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connection attempt {attempt}/{max_retries}...")
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                program=program
            )
            print("✓ Connected successfully!")

            # Test the connection
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%';")
                    banner = cur.fetchone()[0]
                    print(f"\n{banner}")
                except Exception:
                    cur.execute("SELECT 1 FROM DUAL")
                    cur.fetchone()
                    print("  Connected to Oracle Database")

            return conn

        except oracledb.OperationalError as e:
            error_msg = str(e)
            print(f"✗ Connection failed (attempt {attempt}/{max_retries})")

            if "DPY-4011" in error_msg or "Connection reset by peer" in error_msg:
                print("  → This usually means:")
                print("    1. Database is still starting up (wait 2-3 minutes)")
                print("    2. Listener configuration issue")
                print("    3. Container is not running")

                if attempt < max_retries:
                    print(f"\n  Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print("\n  💡 Try running: setup_oracle_database()")
                    print("     This will fix the listener and verify the connection.")
                    raise
            else:
                raise
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            raise

    raise ConnectionError("Failed to connect after all retries")

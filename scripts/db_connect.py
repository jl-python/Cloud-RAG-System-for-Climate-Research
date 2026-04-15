import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Fill .env from .env.example")

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    conn.autocommit = True
    return conn


if __name__ == "__main__":
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT current_user, current_database(), version()")
        print("Connected:", cur.fetchone())
        conn.close()
    except Exception as e:
        print("Connection failed:", e)
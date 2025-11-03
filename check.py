import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv("DATABASE_URL")

conn = psycopg2.connect(database_url, sslmode="require")
cur = conn.cursor()

table_name = "users"

cur.execute("""
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = %s;
""", (table_name,))

columns = cur.fetchall()

print(f"Columns in '{table_name}' table:")
for col in columns:
    name, dtype, nullable, default = col
    print(f"- {name} | {dtype} | Nullable: {nullable} | Default: {default}")

cur.close()
conn.close()

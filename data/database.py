import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST", "localhost")  # default to localhost
DB_PORT = os.getenv("DB_PORT", "5432")       # default port


# Get folder of current script
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
file_path = os.path.join(PROJECT_ROOT, "data", "wealth_evolution.xlsx")
# 1. Connect to default "postgres" database with your macOS user
conn = psycopg2.connect(dbname="postgres", user="nick", host="localhost")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

# 2. Create a new user (role) with password
new_user = DB_USER
new_password = DB_PASS
cur.execute(f"CREATE USER {new_user} WITH PASSWORD '{new_password}';")

# 3. Create a new database owned by this user
new_db = DB_NAME
cur.execute(f"CREATE DATABASE {new_db} OWNER {new_user};")

# 4. Grant all privileges on the database to the user
cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {new_db} TO {new_user};")

cur.close()
conn.close()
print(f"Database '{new_db}' and user '{new_user}' created successfully.")


# pip install python-dotenv sqlalchemy 

from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, Text
from sqlalchemy.orm import declarative_base

# Load environment variables from .env file

# Build connection string
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL, future=True)
Base = declarative_base()

# Define table schema
class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    category = Column(String(100))
    description = Column(Text)
    amount = Column(Numeric(12,2), nullable=False)

# Create table
Base.metadata.create_all(engine)
print(f"Table 'transactions' created in database '{DB_NAME}'.")

# Load Excel
import pandas as pd

df = pd.read_excel(file_path)
df.columns
df["Date"] = pd.to_datetime(df["Date"]).dt.date
df["Amount"] = pd.to_numeric(df["Amount"])

# 6. Insert into DB
df.to_sql("transactions", engine, if_exists="append", index=False, method="multi", chunksize=2000)
print(f"{len(df)} rows imported from Excel into 'transactions'.")
DROP DATABASE postgres

-- Step 1: Create user and database (run as superuser/admin)
-- Connect to default 'postgres' database first

-- Create new user with password
CREATE USER your_db_user WITH PASSWORD 'your_password';

-- Create new database owned by the user
CREATE DATABASE your_db_name OWNER your_db_user;

-- Grant all privileges on the database to the user
GRANT ALL PRIVILEGES ON DATABASE your_db_name TO your_db_user;

-- Step 2: Connect to the new database and create table
-- Switch connection to your new database (your_db_name)

-- Create the transactions table
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    category VARCHAR(100),
    description TEXT,
    amount NUMERIC(12,2) NOT NULL
);

-- Optional: Grant table-level permissions to the user
GRANT ALL PRIVILEGES ON TABLE transactions TO your_db_user;
GRANT USAGE, SELECT ON SEQUENCE transactions_id_seq TO your_db_user;

-- Step 3: Import data from Excel file
-- Note: Direct Excel import in SQL requires specific tools/extensions

-- Option A: If you have converted Excel to CSV, use COPY command:
-- COPY transactions (date, category, description, amount) 
-- FROM '/path/to/your/wealth_evolution.csv' 
-- DELIMITER ',' 
-- CSV HEADER;

-- Option B: Manual INSERT statements (example format):
-- INSERT INTO transactions (date, category, description, amount) VALUES
-- ('2024-01-01', 'Income', 'Salary', 5000.00),
-- ('2024-01-02', 'Expense', 'Groceries', -150.75),
-- ('2024-01-03', 'Investment', 'Stock Purchase', -1000.00);

-- Step 4: Verify the setup
-- Check if table was created successfully
\dt

-- Check table structure
\d transactions

-- Count imported records
SELECT COUNT(*) FROM transactions;

-- Sample data preview
SELECT * FROM transactions LIMIT 5;





import mysql.connector

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",      # or "127.0.0.1"
    user="root",           # your MySQL username
    password="1234", # your MySQL password
    database="waste_management" # database name
)

cursor = db.cursor()
print("✅ Connected to MySQL!")

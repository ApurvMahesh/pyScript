import sqlite3

conn = sqlite3.connect("events.sqlite")

cursor = conn.cursor()
sql_query = """ CREATE TABLE events (
    id integer PRIMARY KEY,
    event_type text NOT NULL,
    event_name text NOT NULL
)"""
cursor.execute(sql_query)
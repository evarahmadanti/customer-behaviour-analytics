import mysql.connector

def connection():
  mydb = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="02_Jan99",
    database="customer_behaviour_analytics"
  )
  return mydb
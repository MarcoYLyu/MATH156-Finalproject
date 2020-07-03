import numpy as np
import pandas as pd
import pickle
import sqlite3

data_dir = "./data/videogames.csv"
database_dir = './data/math156_final.db'
table_name = 'VIDEOGAMES'

def read_data():
    conn = None
    try:
        conn = sqlite3.connect(database=database_dir, user="postgres", password="postgres")
        data = pd.read_csv(data_dir, delimiter=",", encoding="unicode_escape")
        create_table(data.columns)
    
def create_table(column_names, cur):
    command = "CREATE TABLE IF NOT EXISTS {0} (".format(table_name)
    cols = list(map(lambda col: col.lower(), column_names))
    command += ", ".join(cols)
    command += ');'
    cur.execute(command)
    
    
if __name__ == "__main__":
    read_data()
import numpy as np
import pandas as pd
import pickle
import sqlite3
import re

class Videogames(object):
    def __init__(self, database_dir):
        self.database_dir = database_dir
        self.table = ""
        
        self._has_data = False
        self._headers = []
        self._dtypes = []
        self._connection = None
        try:
            with open('./data/data.pickle', "rb") as f:
                self.table, self._headers, self._dtypes, self._has_data = pickle.load(f)
        except:
            pass
        
    @property
    def table_name(self):
        return self.table

    @property
    def status(self):
        return self.get_status()
    
    @property    
    def headers(self):
        return self._headers
    
    @property
    def dtypes(self):
        return self._dtypes
    
    def get_status(self):
        """
        
        Returns:
            Boolean: whether we have loaded the data from csv
        """
        return self._has_data
    
    def get_headers(self):
        return self._headers

    def get_dtypes(self):
        return self._dtypes
    
    def read_data_in(self, filepath, table, write_headers=False):
        conn = sqlite3.connect(database=self.database_dir)
        cur = conn.cursor()
        data = pd.read_csv(filepath, delimiter=",", encoding="unicode_escape")
        
        self.table = table
        headers = self._get_headers(data)
        dtypes = self._get_dtypes(data)
        self._create_table(headers, dtypes, cur)
        
        if write_headers:
            with open('./data/headers.csv', "w+") as f:
                f.write(", \n".join(headers))
        
        if not self._has_data:
            with open("./data/data.pickle", "wb+") as f:
                self._insert_data(data, headers, dtypes, cur)
                pickle.dump((self.table, headers, dtypes, True), f, pickle.HIGHEST_PROTOCOL)
        
        del data
        conn.commit()
        conn.close()
    
    def get_col(self, header):
        if not self._connection:
            self._connection = sqlite3.connect(self.database_dir)
        cur = self._connection.cursor()
        
        command = "SELECT {0} FROM {1};".format(header, self.table)
        return self._col2list(cur.execute(command).fetchall())
    
    def execute(self, command):
        if not self._connection:
            self._connection = sqlite3.connect(self.database_dir)
        cur = self._connection
        
        if bool(re.match("^SELECT", command, re.I)):
            return list(self._col2list(cur.execute(command).fetchall()))
        else:
            print("ILLEGAL COMMAND")


    ## Helper Functions ##  
    def _get_headers(self, data):
        """Return the headers of the data

        Args:
            data DataFrame: the data we read from csv.

        Returns:
            list: the headers of the data
        """
        if not self._headers:
            self._headers = list(map(lambda col: col.lower(), data.columns))
        return self._headers
    
    def _get_dtypes(self, data):
        if not self._dtypes:
            self._dtypes = [self._process_dtype(type(data[col][0])) for col in data.columns]
        return self._dtypes
       
    def _create_table(self, headers, dtypes, cur):
        command = "CREATE TABLE IF NOT EXISTS {0} (".format(self.table)
        template = "{0} {1}"
        n = len(headers)
        for i in range(n):
            command += template.format(headers[i], dtypes[i])
            if i != n - 1:
                command += ", "
        command += ");"
        cur.execute(command)
        
    def _insert_data(self, data, headers, dtypes, cur):
        command_template = "INSERT INTO {0} ({1}) VALUES ({2});"
        
        for i, itr in data.iterrows():
            res = list(map(self._str_classifier, list(itr)))
            command = command_template.format(self.table, ", ".join(headers), self._list2str(res, dtypes))
            cur.execute(command)

    def _list2str(self, data, dtypes):
        res = ""
        for i in range(len(data)):
            if dtypes[i] == "NUMBER":
                if data[i] == "NULL":
                    res += "-1"
                else:
                    res += str(data[i])
            else:
                res += "\"{0}\"".format(data[i])
            if i != len(data) - 1:
                res += ", "
        return res

    def _str_classifier(self, x):
        if type(x) == float and np.isnan(x):
            return "NULL"
        return x

    def _process_dtype(self, dtype):
        type_converter = {type(',') : "VARCHAR(80)", np.float64: "NUMBER"}
        return type_converter[dtype]
    
    def _col2list(self, col):
        n = len(col[0])
        return list(map(lambda x: list(x)[:n], col))
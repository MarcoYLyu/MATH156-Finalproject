import numpy as np
import pandas as pd
import pickle
import sqlite3
import re
import os
from sklearn.impute import SimpleImputer as Imputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

__all__ = ['Videogames']

def get_dir(path):
    return os.path.join(getWorkDir(), path)

def getWorkDir():
    pathlist = os.path.abspath(os.curdir).split('/')
    path = '/'
    for p in pathlist:
        path = os.path.join(path, p)
        if p == 'video-game-sales-predictor' or p == 'video-game-sales-predictor-master':
            break
    return path

class Videogames(object):
    def __init__(self, database_dir, data_dir='data/', storage='data'):
        self.database_dir = database_dir
        self.table = ''
        self.data_dir = data_dir
        self.storage = '{0}.pickle'.format(storage)
        
        self._has_data = False
        self._headers = []
        self._dtypes = []
        self._connection = None
        try:
            with open(get_dir(data_dir + self.storage), "rb") as f:
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
            with open(get_dir(self.data_dir + 'headers.csv'), "w+") as f:
                f.write(", \n".join(headers))
        
        if not self._has_data:
            data = self._remove_missing(data)
            with open(get_dir(self.data_dir + self.storage), "wb+") as f:
                self._insert_data(data, headers, dtypes, cur)
                self._has_data = True
                pickle.dump((self.table, headers, dtypes, True), f, pickle.HIGHEST_PROTOCOL)
        
        del data
        conn.commit()
        conn.close()

    def _remove_missing(self, data):
        data = data.replace(r'tbd', np.nan, regex=True)
        data['User_Score'] = data['User_Score'].astype(np.float64)
        condition = (data['Platform'].notnull() & data['Genre'].notnull() & data['Publisher'].notnull() & data['Year_of_Release'].notnull())
        data = self._imputation(data[condition])
        return data

    def _imputation(self, data):
        imp = Imputer(strategy='median')
        attributes = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']
        for item in attributes:
            data[item] = imp.fit_transform(data[[item]]).ravel()
        return data

    def get_col(self, *header):
        if not self._connection:
            self._connection = sqlite3.connect(self.database_dir)
        cur = self._connection.cursor()
        
        command = "SELECT {0} FROM {1};".format(self._list2str(header), self.table)
        return self._col2list(cur.execute(command).fetchall())
    
    def execute(self, command):
        if not self._connection:
            self._connection = sqlite3.connect(self.database_dir)
        cur = self._connection

        if bool(re.match("^[ \t\n]*SELECT", command, re.I)):
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
            self._dtypes = [self._process_dtype(data[col][0]) for col in data.columns]
        return self._dtypes
       
    def _create_table(self, headers, dtypes, cur):
        """Execute the following SQL command

        CREATE TABLE IF NOT EXISTS {table} (
            name VARCHAR(80),
            ...  
        );
        
        Args:
            headers (list): the list of columns where each header is lowercase.
            dtypes (list): the list of types where each type is either NUMBER or VARCHAR(80) based on this data set.
            cur (sqlite3.connection.cursor): a connection cursor of sqlite3 database
        """
        command = "CREATE TABLE IF NOT EXISTS {0} (".format(self.table)
        template = "{0} {1}"
        
        n = len(headers)
        
        ## Convert the data to suitable form for _list2str function
        data = [template.format(headers[i], dtypes[i]) for i in range(n)]
        
        command += self._list2str(data)
        command += ");"
        cur.execute(command)
        
    def _insert_data(self, data, headers, dtypes, cur):
        command_template = "INSERT INTO {0} ({1}) VALUES ({2});"
        for i, itr in data.iterrows():
            res = list(map(self._str_classifier, list(itr)))
            command = command_template.format(self.table, ", ".join(headers),
                                              self._list2str(res, classify=self._row_classifier(res, dtypes)))
            cur.execute(command)

    def _list2str(self, data, delimiter=",", classify=lambda x: x):
        """Convert the list to a string
        
        I have not found such a function in Python and therefore
        wrote one.

        Args:
            data (list): the row of the table
            delimiter (str, optional): the delimiter.
            classify (function, optional): a function that classifies the data in the row.

        Returns:
            str: a string representing the data converted to a string.
        """
        res = ""
        for i in range(len(data)):
            res += classify(data[i])
            if i != len(data) - 1:
                res += delimiter + " "
        return res
    
    def _row_classifier(self, data, dtypes):
        ### classify the data in a row in the table
        def classifier(x):
            i = data.index(x)
            if dtypes[i] == "NUMBER":
                if x == "NULL" or x == 'tbd':
                    return "-1"
                else:
                    return str(x)
            else:
                return "\"{0}\"".format(x)
        return classifier

    def _str_classifier(self, x):
        ### classify the data so that it does not contain nan
        if type(x) == float and np.isnan(x):
            return -1
        return x

    def _process_dtype(self, var):
        dtype = type(var)
        if dtype == str and var.isnumeric():
            return "NUMBER"
        type_converter = {type(',') : "VARCHAR(80)", np.float64: "NUMBER", np.int64: "NUMBER"}
        return type_converter[dtype]
    
    def _col2list(self, col):
        n = len(col[0])
        return list(map(lambda x: list(x)[:n], col))
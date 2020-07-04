# Documentation

This is the documentation for [helper.py](../helper.py).

## Videogames

### Overview

**Videogames** is a Python class in [helper.py](../helper.py) that provides an interface for data extraction from the [csv file](../../data/videogames.csv) via sqlite3 database. Below are the two main reasons why such an interface is provided rather than reading from the csv file directly.

1. Reading directly from the csv file is not efficient enough when the data set is large.
2. Prevent any unnecessary modification to the database (*insertion* or *deletion*).

### Functions

This section explains certain important functions in this class that might be useful for further data analysis.

* `read_data_in` : This function reads the data from the csv file given the path, creates a table given the name. If `write_headers` is true (*default is* `False`), it will write the headers to a file named `data/headers.csv`.

    ```Python
    read_data_in(self, filepath, table, write_headers=False)
    """
        filepath [str]: the path to the csv file
        table [str]: the name of the table
        write_headers [bool]: whether writing the headers down
    """
    ```

    If we have not read the data before, it will create a file called `data.pickle` and store useful information about the data in it so that when we call the function again, it will not go through the process one more time.

    Note: You must call this function before every other function below.

    ```Python
    >>> vg = Videogames(database_dir)
    >>> vg.read_data_in(data_dir, table_name, True)
    ```

* `get_col` : This function gets the key(s) (header) and returns the corresponding column in a Python list in which each item corresponds to a Python list of features. See below for example.

    ```Python
    >>> vg.get_col("name", "publisher")
    [
        ['Miyako', 'Idea Factory'],
        ['Motto NUGA-CEL!', 'Idea Factory'],
        ['NHL 09', 'Electronic Arts'],
        ['11eyes: CrossOver', '5pb'],
        ...
    ]
    ```

* `execute` : This function will execute `SQL` Select command and returns a list in which each item corresponds to a list of features. See below for example.

    ```Python
    >>> vg.execute("SELECT name, publisher FROM VIDEOGAMES WHERE jp_sales > 6;")
    [
        ['Super Mario Bros.', 'Nintendo'],
        ['Pokemon Red/Pokemon Blue', 'Nintendo'],
        ['New Super Mario Bros.', 'Nintendo'],
        ['Pokemon Gold/Pokemon Silver', 'Nintendo'],
        ['Pokemon Diamond/Pokemon Pearl', 'Nintendo']
    ]
    ```

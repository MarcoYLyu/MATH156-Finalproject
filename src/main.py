from helper import Videogames

data_dir = "./data/videogames.csv"
database_dir = './data/math156_final.db'
table_name = 'VIDEOGAMES'

if __name__ == "__main__":
    vg = Videogames(database_dir)
    vg.read_data_in(data_dir, table_name, True)
    print(vg.execute("SELECT name, publisher FROM {0} WHERE jp_sales > 3;".format(vg.table)))
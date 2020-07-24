import pytest
import os
from src import Videogames

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

data_path = 'tests/test_data/'
headers = ['Name', 'Platform', 'Year_of_Release', 
        'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
        'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count',
        'User_Score', 'User_Count', 'Developer']
na_sales = [41.36, 29.08, 15.68, 15.61]
names_globalsales = [['Wii Sports', 82.53],
                    ['Super Mario Bros.', 40.24],
                    ['Mario Kart Wii', 35.52],
                    ['Wii Sports Resort', 32.77]]

@pytest.fixture
def videogames():
    videogames = Videogames(get_dir(data_path + 'test.db'), data_path, storage='test1')
    videogames.read_data_in(get_dir(data_path + 'sample.csv'), 'VIDEOGAMES', False)
    return videogames

class TestHelper:
    def test_no_data(self):
        vg = Videogames(get_dir(data_path + 'test2.db'), data_path, storage='test2')
        assert not vg.status

    def test_constructor(self, videogames):
        assert videogames.table == 'VIDEOGAMES'
        assert videogames.table_name == 'VIDEOGAMES'
        for x, y in zip(headers, videogames.headers):
            assert x.lower() == y

    def test_get_col(self, videogames):
        sales = videogames.get_col('na_sales')
        name_sales = videogames.get_col('name', 'global_sales')
        for actual, data in zip(na_sales, sales):
            assert actual == data[0]
        for actual, data in zip(names_globalsales, name_sales):
            assert actual[0] == data[0]
            assert actual[1] == data[1]

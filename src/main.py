import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from helper import Videogames

if __name__ == "__main__":
    videogames = Videogames("./data/math156.db")
    videogames.read_data_in("./data/videogames.csv", "VIDEOGAMES", True)
    res = np.array(videogames.execute('''
        SELECT name, jp_total, cscore FROM (
            SELECT name AS name,
                   SUM(JP_sales) AS jp_total,
                   critic_score AS cscore
            FROM VIDEOGAMES 
            WHERE critic_score != -1 and year_of_release >=2010
            GROUP BY name) AS VideogameSummary
        WHERE jp_total != 0.0 and cscore >= 1
        ORDER BY jp_total DESC;
        '''))
    xs = np.array(res[:, 2], dtype=np.float64)
    xs.shape = (len(xs), 1)
    ys = np.array(res[:, 1], dtype=np.float64)

    model = make_pipeline(PolynomialFeatures(3), Ridge())
    model.fit(xs, ys)

    newxs = np.linspace(0, 100, 1000)
    temp = newxs.copy()
    temp.shape = (len(newxs), 1)
    newys = model.predict(temp)

    plt.plot(xs, ys, 'r*', newxs, newys, 'b-')
    plt.savefig('jpsales.png', bboxes_inches='tight')
    print(model.score(xs, ys))
    print(model.predict([[9]]))

    
    
    
# MATH156-Finalproject
* *This repository contains the implementation of: **Predicting the Sale of Video Game**, the final project of Math156 group5 at UCLA. In this project, we will investigate and attempt to model the global sales of video games by using
local sales, critic scores, user scores, and other indicators.*

------------------

## Built With

* [Python3](https://www.python.org/download/releases/3.0/)
* [NumPy](https://numpy.org/doc/stable/reference/index.html)


## Preview
* **Motivation:**
The expansion of the video game industry in recent decades has fueled gamers and game studios
all over the world. Creating a video game is an intensive project that requires a diverse array of
resources that would be very costly for the studio if a release goes awry. As a video game
producer, investor, or consultant, the ability to project global sales is very useful when it comes
to possible translations, global releases, and general marketing investment in other parts of the
world. In an effort to solve this problem, we will apply machine learning techniques to model
global video game sales.

* **Method:**
We plan on attempting [regression analysis](https://en.wikipedia.org/wiki/Regression_analysis) to obtain a model for global game sales. A
justification for using this model is that because our desired output is a continuous quantity,
regression techniques are appropriate. We will study the performances of:  
    * [linear regression](https://en.wikipedia.org/wiki/Linear_regression)  
    * [Random Forest](https://en.wikipedia.org/wiki/Random_forest)  
    * [K-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)   

* **Data set:**
The data set we are going to train our model is [Video Game Sales with Ratings](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings) from [Kaggle](https://www.kaggle.com/). The dataset consists of 11,563 video game titles detailing release year, publisher, platform, genre, regional sales, global sales, critic and user scores, critic and user counts, and ESRB rating. We expect to partition the dataset into training and test data.

* **Evaluation:**
To evaluate the effectiveness and accuracy of our models, we will use [root mean squared error
(RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) as it takes into account negative values and is also a commonly used metric in
determining performance of regression models.

## Authors

* **Kevin Weng, Chen Li, Yi Lyu, Omar Hafez** - *at UCLA* -

## Acknowledgments
* The is the Final Project of Math156 Summer 2020 UCLA.

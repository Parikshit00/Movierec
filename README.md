# Movie-recommendation-system-using-Tensorrec

# What is TensorRec?
TensorRec is a Python recommendation system that allows you to quickly develop recommendation algorithms and customize them using TensorFlow.

TensorRec lets you to customize your recommendation system's representation/embedding functions and loss functions while TensorRec handles the data manipulation, scoring, and ranking to generate recommendations.

A TensorRec system consumes three pieces of data: `user_features`, `item_features`, and `interactions`. It uses this data to learn to make and rank recommendations.

![TensorRec System Diagram](https://raw.githubusercontent.com/jfkirk/tensorrec/master/examples/system_diagram.png)

## Dataset
For this recommender system, I have used the [MovieLens Dataset](https://grouplens.org/datasets/movielens/).
I choose the 100k dataset for less cpu intensive computation. The model can be trained using 25M dataset also.

## Data Pre-processing

  ##### Firstly,
  ratings were loaded from ratings.csv. 
  In raw format it looks like,

  ```
  userId,movieId,rating,timestamp
  1,1,4.0,964982703
  1,3,4.0,964981247
  1,6,4.0,964982224
  1,47,5.0,964983815
  1,50,5.0,964982931
  ```

  Each row represents a single rating : one user’s thoughts about one movie and its timestamp.
  
  ##### Secondly,
  The ratings were split into training and test sets by shuffling and splitting. The prototype will be trained on the training set, and the evaluation of their success is analyzed using the test set.
  
  ##### Lastly,
  These ratings were reorganized in to a `Scipy sparse matrix`. In this matrix, every row represents a user and every column is a movie. The `[i,j]th` value in this matrix is  User i’s interaction with Movie j.






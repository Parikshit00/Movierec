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
  **Output:**
  ```
  ----------------------------------------------------------Loading ratings---------------------------------------------

------------------------------------------------------------------Raw ratings example:

['userId', 'movieId', 'rating', 'timestamp']

[0, 0, 4.0, '964982703']------------------------------------------
```
  
  ##### Secondly,
  The ratings were split into training and test sets by shuffling and splitting. The prototype will be trained on the training set, and the evaluation of their success is analyzed using the test set.
  
  ##### Lastly,
  These ratings were reorganized in to a `Scipy sparse matrix`. In this matrix, every row represents a user and every column is a movie. The `[i,j]th` value in this matrix is  User i’s interaction with Movie j.
  
  
## Collaborative Filtering
Tensorrec by default performs matrix factorization for collaborative filtering. In matrix factorization, two matrices are learnt(user representations and item representations). 
![Matrix Factorization](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/NMF.png/400px-NMF.png)
Here, rows of W are the user representations, columns of H are the item representations, and values in V are the interactions. W and H need to be learned to produce the best approximation of V. V is calculated by multiplying W and H.

#### Performance evaluation for collaborative filtering
The metric `Recall at K` is used to evaluate the performance of the prototype. A recall at 10 value of .04 indicates that there is a 4% chance that a particular movie that someone likes will make it in to their top 10 recommendations.
A rating of at least 4.0 is only counted as liked movie.









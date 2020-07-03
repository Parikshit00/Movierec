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
  **Output:**
  ```
  -------------------------------------------80668 train ratings, 20168 test ratings---------------------------------------
  ```
  
  ##### Lastly,
  These ratings were reorganized in to a `Scipy sparse matrix`. In this matrix, every row represents a user and every column is a movie. The `[i,j]th` value in this matrix is  User i’s interaction with Movie j.
  
  
  
## Collaborative Filtering
Tensorrec by default performs matrix factorization for collaborative filtering. In matrix factorization, two matrices are learnt(user representations and item representations). 
![Matrix Factorization](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/NMF.png/400px-NMF.png)
Here, rows of W are the user representations, columns of H are the item representations, and values in V are the interactions. W and H need to be learned to produce the best approximation of V. V is calculated by multiplying W and H.

#### Performance evaluation for collaborative filtering
The metric `Recall at K` is used to evaluate the performance of the prototype. A recall at 10 value of .04 indicates that there is a 4% chance that a particular movie that someone likes will make it in to their top 10 recommendations.
A rating of at least 4.0 is only counted as liked movie.
**Output:**
```
------------------------------------------------------------Training collaborative filter--------------------------------------------------------
-------------------------------------------------Matrix factorization collaborative filter:-------------------------------------------------------------

----------------------------------------------Recall at 10: Train: 0.0021 Test: 0.0008----------------------------------------------
```
**conclusion:** Only 0.08% chance that a liked movie made it to top 10.

#### Improvising Collaborative filter prototype using loss graph "WMRB(weighted margin-rank batch)"
TensorRec uses RMSE (root mean square error) as the loss graph by default but it is not suitable for our recommender system. WMRB works by taking a random sample of items the user hasn’t interacted with and comparing their predictions to items the user likes. Over time, this pushes items a user likes to the top of the rankings.
`In this case, the model was trained on only the positive ratings (≥4.0) so WMRB pushes those to the top`
**Output:**
``` 
---------------------------------------------------------WMRB matrix factorization collaborative filter:--------------------------------------------------------------------------

----------------------------------------------Recall at 10: Train: 0.1168 Test: 0.0763----------------------------------------------
```
**conclusion:** The performance of Collaborative filter incresed significantly by using WMRB loss graph (from 0.08% chance to 7.6% chance).

## Content Based Filtering

#### First Step: Adding Metadata Features
The movie metadata of the MovieLens dataset was utilized which was present in movies.csv file.
In raw format,
```movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
5,Father of the Bride Part II (1995),Comedy
6,Heat (1995),Action|Crime|Thriller
```
This data is read and the movies were mapped to our internal IDs inorder to keep track of the genres for each movie.
**Output:**
```
------------------------------------------------------------Loading movie metadata------------------------------------------------------------------------

-----------------------------------------------------------------------Raw metadata example:

['movieId', 'title', 'genres']

[0, 'Toy Story (1995)', ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']]--------------------------------------------------------------------
```

#### Second Step: Binarizing the genre labels.
The genre labels were binarized using `Scikit’s MultiLabelBinarizer`. The binarized output will be the features for content based recommender system.
**Output:**
```
-----------------------------------------------------Binarized genres example for movie Toy Story (1995):

[0 0 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0]-------------------------------------------------------------------------------------------------
```

### Third Step: Training the content based recommender and performance evaluation
TensorRec model was configured to use a pass-through graph for item features(Item representation = Item features & user representation = how much the user likes that particular genre).

**Output**
```
--------------------------------------------------------------Training content-based recommender-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------Content-based recommender:----------------------------------------------------------------------------------------------

----------------------------------------------Recall at 10: Train: 0.0492 Test: 0.0117----------------------------------------------
```
**Conclusion:** we get a 1.17% chance that a liked movie made it to top 10.

## Hybrid model
A system that combines collaborative filtering and content-based recommendation is called “hybrid” model. Here, indicator features from a collaborative filter and content features derived from metadata used in content based filtering are utilized. These two set of features are then stacked for the hybrid model.
```
------------------------------------------------------------------------Hybrid recommender:--------------------------------------------------------------------

----------------------------------------------Recall at 10: Train: 0.1232 Test: 0.0854----------------------------------------------
```
**Conclusion:** The hybrid model achieved a recall value of 8.54% which is the best result achieved yet.

## Making actual recommendations :
A user was choosen and all the features of the user were passed into the recommender system to get personalized recommendation for the particular user.

**User 432's training data:**
```
_________________________________________

-------------------------------------------------------------------------User 432 had already liked:-------------------------------------------------------

__________________________________________

Usual Suspects, The (1995)

Pulp Fiction (1994)

Silence of the Lambs, The (1991)

Fargo (1996)

Reservoir Dogs (1992)

American History X (1998)

Fight Club (1999)

Shawshank Redemption, The (1994)

Dark Knight Rises, The (2012)

Trainspotting (1996)

Arrival (2016)

My Big Fat Greek Wedding (2002)
```
**Recommendation for user 432:**
```
______________________________________________________________________________

-----------------------------------------------------Recommended movies for User 432 :------------------------------------------------------

__________________________________________________________________________________

Seven (a.k.a. Se7en) (1995)

Braveheart (1995)

Pulp Fiction (1994)

Silence of the Lambs, The (1991)

Saving Private Ryan (1998)

Matrix, The (1999)

Fight Club (1999)

Shawshank Redemption, The (1994)

Memento (2000)

Godfather, The (1972)

```

**User 432's test datas:**
```
______________________________________________________________________________

-------------------------------------------------Some of User 432's held-out movies (movies liked by user present in test dataset of our model)-------------------------------------------------------

______________________________________________________________________________

Kill Bill: Vol. 1 (2003)

Social Network, The (2010)
```










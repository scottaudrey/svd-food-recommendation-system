import numpy as np

def split_ratings(ratings, test_frac=0.10, seed=0):
  rng = np.random.default_rng(seed)
  is_test = np.zeros(len(ratings), dtype=bool)

  user_left = ratings.groupby("user_id").size().to_dict()
  item_left = ratings.groupby("recipe_code").size().to_dict()

  for pos in rng.permutation(len(ratings)):  # goes through the ratings randomly
    row = ratings.iloc[pos]
    user_id, recipe_code = row["user_id"], row["recipe_code"]

    if (rng.random() < test_frac   # each of the ratings now has a 10% chance of being the test rating
      and user_left[user_id] > 1
      and item_left[recipe_code] > 1):
      is_test[pos] = True
      user_left[user_id] -= 1
      item_left[recipe_code] -= 1

  return ratings.iloc[~is_test].copy(), ratings.iloc[is_test].copy()  # returns the integer

N_USERS, N_ITEMS = len(user_ids), len(recipe_ids)

def to_matrix(df):
    R = np.full((N_USERS, N_ITEMS), np.nan)
    u = user_ids.get_indexer(df["user_id"])
    i = recipe_ids.get_indexer(df["recipe_code"])
    R[u, i] = df["stars"].to_numpy()
    return R

def rmse(pred, truth):
  ## Root mean squared error, measured in starts.
  return np.sqrt(np.mean((pred - truth)**2))

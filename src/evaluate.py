import numpy as np
import pandas as pd
import os, urllib.request, zipfile

url = "https://github.com/ryannewhouse11/FoodReviewData/releases/latest/download/recipereview.zip"
urllib.request.urlretrieve(url, "/content/recipereview.zip")

with zipfile.ZipFile("/content/recipereview.zip") as z:
  csv_files = [
      name for name in z.namelist()
      if name.lower().endswith(".csv") and not name.startswith("_MACOSX/")
  ]
  with z.open(csv_files[0]) as f:
    ratings = pd.read_csv(
        f,
        sep=",",
        header=0,
        names=["num_records", "recipe_number", "recipe_code", "recipe_name",
               "comment_id", "user_id", "user_name", "user_reputation", "created_at",
               "reply_count", "thumbs_up", "thumbs_down", "stars", "best_score", "text"]
    )



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

train, test = split_ratings(ratings)
user_ids = pd.Index(train["user_id"].unique())
recipe_ids = pd.Index(train["recipe_code"].unique())
N_USERS, N_ITEMS = len(user_ids), len(recipe_ids)

def to_matrix(df):
  n_users = int(df["user_id"].max())
  n_items = int(df["recipe_code"].max())
  R = np.full((n_users, n_items), np.nan)
  u = df["user_id"].to_numpy(dtype=int) - 1
  i = df["recipe_code"].to_numpy(dtype=int) - 1
  R[u, i] = df["stars"].to_numpy(dtype=float)
  return R

def rmse(pred, truth):
  ## Root mean squared error, measured in starts.
  return np.sqrt(np.mean((pred - truth)**2))

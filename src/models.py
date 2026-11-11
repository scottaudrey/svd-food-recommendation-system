import numpy as np

def bias_baseline(R):
    """Overall mean plus one user effect and one recipe effect."""
    known = ~np.isnan(R)
    mu = np.nanmean(R)

    n_user = known.sum(axis=1)
    b_user = np.nanmean(R - mu, axis=1)

    has_user_rating = n_user > 0

    b_user[has_user_rating] = (np.nansum(R - mu, axis=1)[has_user_rating]
        / n_user[has_user_rating]
    )

    residual = R - mu - b_user[:, None]

    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])

    has_item_rating = n_item > 0

     b_item[has_item_rating] = (
        np.nansum(residual, axis=0)[has_item_rating]
        / n_item[has_item_rating]
    )

    return mu, b_user, b_item

def baseline_matrix(R):
  """Full matrix: overall mean plus user and recipe effects."""
  mu, b_user, b_item = bias_baseline(R)
  return mu + b_user[:, None] + b_item[None, :]

def rank_k(U, s, Vt, k):
  """Build a rank-k matrix from an already computed SVD."""
  return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def iterative_svd_predict(R, k, n_iter=10):
  """Predict ratings by repeatedly"""
  known = ~np.isnan(R)
  B = baseline_matrix(R)
  X = np.where(known, R - B, 0.0)

  for _ in range(n_iter):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    L = rank_k(U, s, Vt, k)
    X = np.where(known, R - B, L)

  return L + B

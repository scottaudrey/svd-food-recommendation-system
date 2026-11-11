def baseline_matrix(R):
  """Full matrix: overall mean plus user and recipe effects."""
  mu, b_user, b_item = bias_baseline(R)
  return mu + b_user[:, None] + b_item[None, :]

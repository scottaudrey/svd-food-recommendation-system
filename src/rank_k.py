def rank_k(U, s, Vt, k):
  """Build a rank-k matrix from an already computed SVD."""
  return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

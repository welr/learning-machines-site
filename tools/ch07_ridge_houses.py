"""Data for the Chapter 7 interactive figure "Ridge, one direction at a time".

The book's shrinkage example (Section 7.2) trades floor area against number of
rooms at about 25 square meters per room, but gives no data set. These twelve
houses supply one: true price 4 thousand euros per square meter and nothing per
room, rooms = area/25 rounded with noise, price noise sd 30 thousand.

Seed 1971 was chosen from 2,000 seeds because least squares lands on large
coefficients of opposite sign (7.7 per m^2, -87 per room) while the truth is
(4, 0), the book's "compensating coefficients". Prints the arrays the figure
hard-codes and the numbers its notes quote.

    python3 tools/ch07_ridge_houses.py
"""
import numpy as np

SEED, M = 1971, 12
rng = np.random.default_rng(SEED)
area = np.round(rng.uniform(60, 160, M), 0)
rooms = np.round(area / 25 + rng.normal(0, 0.35, M))
price = np.round(4 * area + rng.normal(0, 30, M), 0)

print("area  =", area.astype(int).tolist())
print("rooms =", rooms.astype(int).tolist())
print("price =", price.astype(int).tolist())

# Standardize (population sd), center the response, as the book does before ridge.
sa, sr = area.std(), rooms.std()
Z = np.c_[(area - area.mean()) / sa, (rooms - rooms.mean()) / sr]
yc = price - price.mean()
G = Z.T @ Z
ev = np.linalg.eigvalsh(G)
print(f"X^T X (standardized) = {G.round(3).tolist()};  eigenvalues {ev.round(3).tolist()}")
print(f"m*lambda equals the small eigenvalue at lambda = {ev[0] / M:.4f}")
for lam in (0.0, ev[0] / M, 0.1, 1.0):
    th = np.linalg.solve(G + M * lam * np.eye(2), Z.T @ yc)
    print(f"lambda {lam:6.4f}: {th[0] / sa:6.2f} per m^2, {th[1] / sr:7.1f} per room")

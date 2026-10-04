"""Data for the ch02_01 (polynomial regression) page.

Fifteen training houses and forty test houses from the model behind the book's
Chapter 7 polynomial figures: sizes uniform on [40, 150] m^2, price
160 + (260/110)(x - 40) + 0.02 (x - 40)(x - 150) thousand euros plus Gaussian
noise with sd 25, everything rounded to 0.1.

Seed 1768 was chosen from 3,000 so that the page's lessons all show on one
draw: degree 2 has the lowest test error, training error falls with degree, the
degree-14 fit interpolates in a well-conditioned basis, and that interpolant
dips below zero *between* training houses (to about -280 near 85 m^2), not
only beyond the data. Prints the arrays the page's cells hard-code.

    python3 tools/ch02_01_houses.py
"""
import numpy as np

SEED = 1768
rng = np.random.default_rng(SEED)


def truth(x):
    return 160 + (260 / 110) * (x - 40) + 0.02 * (x - 40) * (x - 150)


x = np.round(np.sort(rng.uniform(40, 150, 15)), 1)
y = np.round(truth(x) + rng.normal(0, 25, 15), 1)
x_test = np.round(np.sort(rng.uniform(40, 150, 40)), 1)
y_test = np.round(truth(x_test) + rng.normal(0, 25, 40), 1)

for name, a in (("x", x), ("y", y), ("x_test", x_test), ("y_test", y_test)):
    print(f"{name} = {a.tolist()}")

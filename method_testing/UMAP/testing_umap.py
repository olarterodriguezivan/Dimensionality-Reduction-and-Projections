import sklearn.datasets
from sklearn.datasets import load_digits

digits = load_digits().data

from time import perf_counter


init_time = perf_counter()

print("Starting loading umap")
import umap

end_time = perf_counter()

print("Loading time", end_time-init_time)

reducer = umap.UMAP()
embedding = reducer.fit_transform(digits)

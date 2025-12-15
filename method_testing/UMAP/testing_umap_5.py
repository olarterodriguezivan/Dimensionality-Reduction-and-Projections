import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import sklearn.datasets
import umap
import umap.plot


data, labels = sklearn.datasets.fetch_openml('mnist_784', version=1, return_X_y=True)


mapper = umap.UMAP(random_state=42).fit(data )


umap.plot.points(mapper, labels=labels.astype(int), theme='fire')
plt.show()


corners = np.array([[-5,-10],
                   [-7,6],
                   [2,-8],
                   [12,4]])

test_points = np.array([
    (corners[0]*(1-x) + corners[1]*x)*(1-y) + 
    (corners[2]*(1-x) + corners[3]*x)*y
    for x in np.linspace(0,1,10)    
    for y in np.linspace(0,1,10)
])


inv_map = mapper.inverse_transform(test_points)



fig = plt.figure(figsize=(12,6))
gs = GridSpec(10, 20, figure=fig)

scatter_ax = fig.add_subplot(gs[:, :10])
digit_ax = np.zeros((10,10), dtype=object)

for i in range(10):
    for j in range(10):
        digit_ax[i,j] = fig.add_subplot(gs[i, 10 + j])


scatter_ax.scatter(mapper.embedding_[:,0], mapper.embedding_[:,1], c=labels.astype(int), cmap='Spectral', s=0.1)

scatter_ax.set(xticks=[], yticks=[])


scatter_ax.scatter(test_points[:,0], test_points[:,1], c='k', s=15)

for i in range(10):
    for j in range(10):
        digit_ax[i,j].imshow(inv_map[i*10 + j].reshape(28,28), cmap='gray')
        digit_ax[i,j].set(xticks=[], yticks=[])
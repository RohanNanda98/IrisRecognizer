import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
(x_train, x_test, y_train, y_test) = \
    train_test_split(iris_dataset['data'], iris_dataset['target'],
                     random_state=0)

iris_dataframe = pd.DataFrame(x_train,
                              columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8,
    cmap=mglearn.cm3,
    )

plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print 'Test set score: {:.2f}'.format(np.mean(y_pred == y_test))

import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = "/mnt/hgfs/ubuntu14/dataset/iris/iris.csv"
classify = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
colors = ['navy', 'turquoise', 'darkorange']
n_classes = 3

iris = datasets.load_iris()
train_data,test_data,train_target,test_target = train_test_split(iris.data,iris.target,test_size=0.25,random_state=0)
if __name__=="__main__":
    means_init = np.array([train_data[train_target == i].mean(axis=0)
                                    for i in range(n_classes)])
    kmeans = KMeans(init=means_init, n_clusters=3, n_init=10)
    kmeans.fit(train_data)

    train_pred = kmeans.predict(train_data)
    train_acc = np.mean(train_pred==train_target)
    print "train accuracy is {}".format(train_acc)
    
    
    test_pred = kmeans.predict(test_data)
    test_acc = np.mean(test_pred==test_target)
    print "test accuracy is {}".format(test_acc)
    
    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 2], data[:, 3], s=0.8, color=color,
                    label=iris.target_names[n])
        plt.scatter(kmeans.cluster_centers_[n,2],kmeans.cluster_centers_[n,3],marker="X",color=color)
    
    plt.text(0.9, 2.3, 'Train accuracy: %.4f' % train_acc)
    plt.text(0.9, 2.1, 'Test accuracy: %.4f' % test_acc)
    
    plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))

    plt.show()
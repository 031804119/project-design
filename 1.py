from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt  
import numpy as np  
import random

fig=plt.figure(1)  
x1,y1=make_circles(n_samples=400,factor=0.5,noise=0.05)
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(x1,y1)

x2=random.random()
y2=random.random()

X_sample=np.array([[x2,y2]])
y_sample=knn.predict(X_sample)
neighbors=knn.kneighbors(X_sample,return_distance=False)

a2=[]
for i in y1:
    if i==0:
        a2.append('b')
    else:
        a2.append('g')

plt.subplot(121)  
plt.title('make_circles 1')
plt.scatter(x1[:,0],x1[:,1],marker='o',c=a2)
plt.scatter(x2,y2,marker='*',c='r')

plt.subplot(122)
plt.title('make_circles 2')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=a2)
plt.scatter(x2, y2, marker='*', c='r')
for i in neighbors[0]:
    plt.scatter(x1[i][0],x1[i][1],  marker='o', c='r')


plt.show()


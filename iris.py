
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np 
iris= load_iris()
test_idx = [0, 50, 100]


''' print iris.feature_names
print iris.target_names
print iris.data[1:3]
print iris.target[1:3]
for i in range (len(iris.target)):
	print"Example %d : label %s, feature %s " % (i, iris.target[i], iris.data[i]) 
'''

# train data

train_target = np.delete(iris.target, test_idx)
train_data= np.delete (iris.data, test_idx, axis=0)

# testing data

test_target = iris.target[test_idx] 
test_data = iris.data[test_idx]


# classify

clf= tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

'''
print test_target
print clf.predict (test_data)
'''

print test_data [2], test_target [2]
print iris.feature_names
print iris.target_names



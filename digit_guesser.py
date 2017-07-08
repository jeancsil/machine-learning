import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.0001, C=100)

training_set = digits.data[:-10]
labels = digits.target[:-10]

x, y = training_set, labels
clf.fit(x, y)

for i in range(10):
    print("Prediction: {}".format(clf.predict([digits.data[-i]])))
    print("Digit:      [{}]".format(digits.target[-i]))

# print('Prediction: ',  clf.predict([digits.data[-1]]))
# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

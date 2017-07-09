from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.0001, C=100)

training_set = digits.data[:-10]
training_labels = digits.target[:-10]

testing_set = digits.data[-10:]
testing_labels = digits.target[-10:]

x, y = training_set, training_labels
clf.fit(x, y)

for i in range(10):
    print("Test set: {}. Predicted: {}".format(testing_labels[i], clf.predict([testing_set[i]])[0]))

import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

database = datasets.load_digits()
print("databasesize:", len(database.images))

clf = svm.SVC(gamma=0.001)
x,y = database.data[:1000], database.target[:1000]
clf.fit(x, y)

expected = database.target[1000:]
predicted = clf.predict(database.data[1000:])

img_predict = list(zip(database.images[1000:], predicted))
for i, (image, prediction) in enumerate(img_predict[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
    plt.title('Prediction:%i' % prediction)

plt.show()

print("classifier report\n%s\n" % (metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

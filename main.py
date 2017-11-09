import numpy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from kernels import LinearKernel, GaussianKernel
from kernel_pca import KernelPCA
from kmeans import Kmeans

cats = [ 'sci.med', 'misc.forsale', 'soc.religion.christian']
newsgroups_all = fetch_20newsgroups(subset='all', categories=cats)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_all.data)
X = vectors.toarray()
y = newsgroups_all.target
X_train = X[0:2000, :]
X_test = X[2000:, :]
y_train = y[0:2000]
y_test = y[2000:]

kpca = KernelPCA(GaussianKernel(sigma=1)) # to change
kpca.fit(X_train)
n_components = None # to change
X_train_proj = kpca.predict(X_train, components=n_components)
X_test_proj = kpca.predict(X_test, components=n_components)

permuts = numpy.array([[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]])

def find_permut_for_prediction(y_pred, y, permuts):
    y_pred_best = y_pred
    accuracy_best = 0.
    permut_best = permuts[0,:]
    for i in range(0, len(permuts)):
        y_pred_current = [permuts[i,e] for e in y_pred]
        accuracy = 1.0 * numpy.sum(numpy.equal(y_pred_current, y)) / len(y)
        if accuracy > accuracy_best:
            accuracy_best = accuracy
            y_pred_best = y_pred_current
            permut_best = permuts[i,:]
    return y_pred_best, accuracy_best, permut_best

def apply_permut_to_prediction(y_pred, y, permut):
    y_pred_new = [permut[e] for e in y_pred]
    accuracy = 1.0 * numpy.sum(numpy.equal(y_pred_new, y)) / len(y)
    return y_pred_new, accuracy

n_iter = 5
kmeans = Kmeans(nclusters=3)
accuracy_test_best_best = 0.
y_pred_train_best_best = numpy.zeros(y_train.shape)
y_pred_test_best_best = numpy.zeros(y_test.shape)
print("Trying different K-means initializations")
for i in range(0, n_iter):
    print("K-means Iteration%d" % i)
    kmeans.fit(data=X_train_proj, niter=30)
    y_pred_train = kmeans.predict(X_train_proj)
    y_pred_test = kmeans.predict(X_test_proj)
    
    y_pred_train_best, accuracy_train_best, permut_train_best = \
        find_permut_for_prediction(y_pred_train, y_train, permuts)
    y_pred_test_best, accuracy_test_best = \
        apply_permut_to_prediction(y_pred_test, y_test, permut_train_best)

    if accuracy_test_best > accuracy_test_best_best:
        accuracy_test_best_best = accuracy_test_best
        y_pred_train_best_best = y_pred_train_best
        y_pred_test_best_best = y_pred_test_best

    print "%.5f" % accuracy_train_best
    print "%.5f" % accuracy_test_best

print "%.5f" % accuracy_test_best_best

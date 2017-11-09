import matplotlib.pyplot as plt
import numpy
from scipy import linalg

class KernelPCA:
    def __init__(self, kernel):
        self.kernel = kernel
        self.X = None
        self.alpha = None
        self.center_vector = None

    def _center(self, X):
        n = X.shape[0]
        U = numpy.ones((n, n)) / n
        K = self.kernel.build_K(X)
        self.center_vector = numpy.mean(K, axis=1) - numpy.mean(K)
        return numpy.dot(numpy.dot(numpy.eye(n) - U, K), numpy.eye(n) - U)

    def fit(self, X):
        print("Fit KPCA")
        n = X.shape[0]
        self.X = X
        K = self._center(X)
        print("Calculating eigenvalue decomposition")
        eig_values, eig_vectors = linalg.eigh(K)

        # eigenvalues in decreasing order
        index = range(n)[::-1]
        eig_values = eig_values[index]
        eig_vectors = eig_vectors[:, index]

        # filter non-positive eigenvalues
        index = eig_values > 0
        eig_values = eig_values[index]
        eig_vectors = eig_vectors[:, index]

        self.alpha = eig_vectors
        for i, v in enumerate(eig_values):
            self.alpha[:, i] /= numpy.sqrt(v)

    def predict(self, X, components=None):
        assert components is None or (components > 0 and components <= self.alpha.shape[1])
        if components is None:
            components = self.alpha.shape[1]

        K = self.kernel.build_K(X, self.X)
        K = K - numpy.mean(K, axis=1)[:, numpy.newaxis] - self.center_vector
        return numpy.dot(K, self.alpha[:, :components])

if __name__ == '__main__':
    from sklearn.datasets import make_circles
    from kernels import LinearKernel, GaussianKernel

    f, axarr = plt.subplots(2, 2, sharex=True)

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    axarr[0, 0].scatter(X[y==0, 0], X[y==0, 1], color='red')
    axarr[0, 0].scatter(X[y==1, 0], X[y==1, 1], color='blue')

    kpca = KernelPCA(LinearKernel())
    kpca.fit(X)
    Xproj = kpca.predict(X)
    axarr[0, 1].scatter(Xproj[y==0, 0], numpy.zeros(500), color='red')
    axarr[0, 1].scatter(Xproj[y==1, 0], numpy.zeros(500), color='blue')

    # decrease sigma to improve separation
    kpca = KernelPCA(GaussianKernel(0.686))
    kpca.fit(X)
    print kpca.alpha.shape[1]
    Xproj = kpca.predict(X)
    axarr[1, 0].scatter(Xproj[y==0, 0], numpy.zeros(500), color='red')
    axarr[1, 0].scatter(Xproj[y==1, 0], numpy.zeros(500), color='blue')

    axarr[1, 1].scatter(Xproj[y==0, 0], Xproj[y==0, 1], color='red')
    axarr[1, 1].scatter(Xproj[y==1, 0], Xproj[y==1, 1], color='blue')

    plt.show()

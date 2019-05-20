import numpy as np


class radial_with_cv(): #Radial Basis Function with Cross Valitation
    def __init__(self,X,Y):
        """
        :type X: np.array
        :type Y: np.array
        """
        self.X = X
        self.Y = Y
        self.d = X.shape[1]
        self.N = X.shape[0]
        self.r = (self.N**(1/(2*self.d)))**(-1)

    def predict(self, x):
        s = 0
        foo = x - self.X[5,:]
        foobar = np.linalg.norm(foo)
        bar = self.phi(0)
        phi = [self.phi(np.linalg.norm(x - self.X[n,:])/self.r) for n in range(self.N)]
        sum_phi = sum(phi)
        for n in range(self.N):
            s += self.wn(n, sum_phi) * phi[n]
        return s

    def phi(self, x):
        return (2*np.pi)**(-self.d/2) * np.exp(-0.5 * x**2)


    def wn(self, n, sum_phi):
        return self.Y[n]/sum_phi
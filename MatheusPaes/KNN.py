import random
import matplotlib.pyplot as plt
import numpy as np

class knn_model:
    def __init__(self, k, data):
        random.shuffle(data)
        self.data = data
        self.data_dim = len(data[0])
        self.k = k
    def change_k(self,k):
        self.k = k
    def dist(self,x,y):
        sq = 0
        for i in range(0,len(x)):
            sq += (x[i]-y[i])**2
        return sq**0.5
    def predict(self,x):
        dists = []
        for d in self.data:
            dists.append((d,self.dist(x,d[0:-1])))
        dists.sort(key=lambda x:x[1])
        knn = dists[0:self.k]
        y = 0
        for n,d in knn:
            y += n[-1]
        return y/self.k


def test(k):
    f = lambda x:x**3
    t = [(i,f(i)) for i in [random.uniform(-10,10) for j in range(0,20)]]

    m = knn_model(k,t)

    x = np.linspace(-11,11,500)
    y = [m.predict((p,)) for p in x]

    x_d = [p[0:-1] for p in t]
    y_d = [p[-1] for p in t]

    plt.plot(x,y)
    plt.plot(x_d,y_d,'ro')
    plt.show()

test(3)

def condensate(k,data,st,e):
    m = knn_model(k,data)
    dt = data.copy()
    random.shuffle(dt)
    pts = dt[0:st]
    for p in pts:
        dt.remove(p)
    
    pts_m = knn_model(k,pts)
    pts_preds = [pts_m.predict(p[0:-1]) for p in data]
    orig_pred = [m.predict(p[0:-1]) for p in data]
    errors = [abs((pred-orig)/orig) for pred,orig in zip(pts_preds,orig_pred)]
    w_pt = -1
    for i in range(0,len(errors)):
        if errors[i] > e:
            w_pt = i
            break
    while not w_pt == -1:
        if len(dt) == 0:
            break
        min_d = m.dist(data[w_pt][0:-1],dt[0][0:-1])
        min_p = data[0]
        for p in dt:
            dist = m.dist(data[w_pt][0:-1],p[0:-1])
            if dist < min_d:
                min_d = dist
                min_p = p
        pts.append(p)
        dt.remove(p)
        pts_m = knn_model(k,pts)
        pts_preds = [pts_m.predict(p[0:-1]) for p in data]
        orig_pred = [m.predict(p[0:-1]) for p in data]
        errors = [abs((pred-orig)/orig) for pred,orig in zip(pts_preds,orig_pred)]
        w_pt = -1
        for i in range(0,len(errors)):
            if errors[i] > e:
                w_pt = i
                break
    return pts,errors

def test_cond(k,e):
    f = lambda x:x**2
    t_pre = [(i,f(i)) for i in [random.uniform(-10,10) for j in range(0,20)]]
    t,errors = condensate(k,t_pre,1,e)
    print(errors)
    m = knn_model(k,t)
    m_pre = knn_model(k,t_pre)

    x = np.linspace(-11,11,500)
    y = [m.predict((p,)) for p in x]
    
    x_pre = np.linspace(-11,11,500)
    y_pre = [m_pre.predict((p,)) for p in x]

    x_d = [p[0:-1] for p in t]
    y_d = [p[-1] for p in t]
    
    x_d_pre = [p[0:-1] for p in t_pre]
    y_d_pre = [p[-1] for p in t_pre]

    plt.plot(x,y,'g')
    plt.plot(x_pre,y_pre,'r--')
    plt.plot(x_d,y_d,'go')
    plt.plot(x_d_pre,y_d_pre,'rx')
    plt.show()

test_cond(2,0.8)


import kNN, pandas as pd
from RBFwCV import radial_with_cv

#######################K-NN

def getArray(path):
    df = pd.read_csv(path)  # type: pd.DataFrame
    df["Salary"] = df["ConvertedSalary"].apply(lambda x: int(round(x)))
    df = df[["AnosProgramando", "Salary", "Gender"]]
    arr = df.to_numpy()
    return arr


def generate_data(path):
    arr = getArray(path)
    list = []
    for lin in range(arr.shape[0]):
        data = kNN.data(arr[lin,0:3])
        list.append(data)
    return list

def learn(list, k):
    counter = 0
    size = len(list)
    first = list.pop(0)
    first.classify(k, list)
    list.append(first)
    elected = list.pop(0)
    while counter<size:
        counter += 1
        elected.classify(k, list)
        list.append(elected)
        elected = list.pop(0)
    return list

def printk(list, k):
    """

    :type list: List<data>
    """
    print("x: " + str(list[k].x))
    print("y: " + str(list[k].y))
    print("predicted: " + str(list[k].guess))

# print("kNN:")
# list = learn(generate_data("GenderCoding.csv"),3)
# printk(list, 12)
# printk(list,15)
# printk(list,18)
# printk(list,27)


################ Radial Basis Function with Cross Validation

def printRBF(predictor, k):
    """

    :type predictor: RBFwCV
    """
    print("x: " + str(predictor.X[k]))
    print("y: " + str(predictor.Y[k]))
    print("predicted: " + str(predictor.predict(X[k])))
arr = getArray("GenderCoding.csv")
X = arr[:,0:2]
Y = arr[:,2]
my_predictor = radial_with_cv(X, Y)

print("Radial Basis Function with Cross Validation")
printRBF(my_predictor, 2)
printRBF(my_predictor, 4)
printRBF(my_predictor, 5)
printRBF(my_predictor, 26)
printRBF(my_predictor, 30)
printRBF(my_predictor, 50)



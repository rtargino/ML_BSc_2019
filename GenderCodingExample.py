import kNN, pandas as pd

def generate_data(path):
    df = pd.read_csv(path) # type: pd.DataFrame
    df["Salary"] = df["ConvertedSalary"].apply(lambda x: int(round(x)))
    df = df[["AnosProgramando", "Salary", "Gender"]]
    arr = df.to_numpy()
    list = []
    for lin in range(arr.shape[0]):
        data = kNN.data(arr[lin,0:2])
        list.append(data)
    return list

def learn(list, k):
    first = list.pop(0)
    first.classify(k, list)
    list.append(first)
    elected = list.pop(0)
    while elected != first:
        elected.classify(k, list)
        list.append(elected)
        elected = list.pop(0)

print(learn(generate_data("GenderCoding.csv"),3))
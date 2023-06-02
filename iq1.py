import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calculate(pred, true):
    return np.sqrt(((pred-true)**2).mean())
if __name__ == '__main__':
    df = pd.read_csv("./data/iq.tsv", sep='\t')
    print(df)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    x_train = train.drop("PIQ", axis=1)
    #x_train = x_train.drop("Height", axis=1)
    #X = [[i, j, k] for i, j, k in zip(train['Brain'], train['Height'], train['Weight'])]
    print(x_train)
    #normalizujemo X
    #ovdje su vrijednosti slicne i nema potrebe za skaliranje
    st = StandardScaler()
    st.fit(x_train)
    #x_train[x_train.columns] = st.transform(x_train[x_train.columns])
    print("++++++")
    print(x_train)
    #X[x_train.columns] = st.transform(x_train[x_train.columns])
    X1 = [[i, j, k] for i, j, k in zip(test['Brain'], test['Height'], test['Weight'])]
    lm = LinearRegression().fit(x_train, train['PIQ'])
    print("Coefficients:", lm.coef_)
    y_pred = lm.predict(X1)
    RMSE = calculate(y_pred, test['PIQ'])
    print(RMSE)

''''
    X = [[i, j, k] for i, j, k in zip(x1, x2, x3)]
    lm = linear_model.LinearRegression()
    #mozemo da proslijedimo kada imamo x koje ima vise parametara
    lm.fit(X, y) #radimo sada zapravo dobijanje nagiba i presjecanja, dobili smo zapravo model
    print("Coefficients:", lm.coef_)
    plt.scatter(y, lm.predict(X))   #ovo je zapravo nasa predvidnje vrijednost
    plt.xlabel("Actual IQ")
    plt.ylabel("Predicted IQ")
    plt.show()'''
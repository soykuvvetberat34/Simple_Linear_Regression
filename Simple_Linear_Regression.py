#@BeratSOYKUVVET
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#Data preprocessing -prepare for simple linear regression
data=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\Tahmin\\aylara_gore_satis.csv")
df=pd.DataFrame(data)
months=df.iloc[:,0]
sales=df.iloc[:,1]
x_train,x_test,y_train,y_test=train_test_split(months,sales,test_size=0.33,random_state=0)
SC=StandardScaler()
x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)
y_train=y_train.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)


#Simple Linear Regressin -> basit doğrusal regresyon 
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)# X_train ve Y_train verilerini kullanarak modeli eğit
#X_train ve Y_train verileri arasındaki en uygun doğrusal ilişkiyi bulmaya çalışır.

#predict (tahmin) algorithms
predict=model.predict(x_test)
#modelimizi yukarıda train data setleri ile eğittik şimdi X_test(aylar) data setini kullanarak eğitildiği 
#üzere satışları(y_test) tahmin etmeye çalışacak
y_test_len=len(y_test)
for i in range(y_test_len):
    print(f"gerçek satış verisi: {y_test[i]} tahmin edilen: {predict[i]}")
    i+=1
    
x_train_df=pd.DataFrame(x_train)
y_train_df=pd.DataFrame(y_train)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='g',label="Eğitim verileri")
plt.scatter(x_test,y_test,color='b',label="Test verileri")
plt.plot(months,model.predict(months.values.reshape(-1,1)),color='r',label="Tahmin edilen satışlar")
plt.title("Basit Doğrusal Regresyon")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.legend(loc="upper left")
plt.show()


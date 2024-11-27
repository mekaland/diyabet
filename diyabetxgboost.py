from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

#datayı yükledik
data = pd.read_csv("C:/Users/hp/Downloads/diabetes (2).csv" )
veri = data.copy()

y = data["Outcome"]
X= data.drop(columns="Outcome",axis=1)

#  Verinin eğitim ve test setlerine ayrılması
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.2,random_state=42)

# Verilerin ölçeklendirilmesi
x = StandardScaler()
X_train = x.fit_transform(X_train)
X_test = x.transform(X_test)

# Model kurma
modelxgb = XGBClassifier(learning_rate=0.2,max_depth=3,n_estimators=500,subsample=0.7)
modelxgb.fit(X_train,y_train)
tahminxgb = modelxgb.predict(X_test)

modelbay = GaussianNB()
modelbay.fit(X_train,y_train)
tahminbay = modelbay.predict(X_test)

# Model doğruluğunun hesaplanması
acs= accuracy_score(y_test,tahminxgb)
print(acs*100)

acs2= accuracy_score(y_test,tahminbay)
print(acs2*100)

#parametreler = {"max_depth":[3,5,7],    
#"subsample":[0.2,0.5,0.7],
#"n_estimators":[500,1000,2000],
#"learning_rate":[0.2,0.5,0.7]}

#grid = GridSearchCV(modelxgb,param_grid=parametreler,cv=10,n_jobs=-1)
#grid.fit(X_train,y_train)
#print(grid.best_params_)
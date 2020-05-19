import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

style.use('ggplot')

df = pd.read_excel('titanic.xls')
#print(df.head())

df.drop(['body','name'],1,inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0,inplace=True)
#print(df.head())
#handeling nonnumericdata
#df.drop(['sex'],1,inplace=True)
def handle_non_numeric_data(df):
    columns = df.columns.values
    print(columns)
    for column in columns:
        text_digit_vals={}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            #print(unique_elements)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    #print(text_digit_vals)
                    x = x+1
            df[column] = list(map(convert_to_int, df[column]))

    return df






df=handle_non_numeric_data(df)
print(df.head())

X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)
y=np.array(df['survived'])

clf=KMeans(n_clusters=2)
clf.fit(X)

correct=0
#print(np.array(X[0].astype(float)))
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction =clf.predict(predict_me)
    #print(prediction)
    if prediction == y[i]:
        correct+=1


print(correct/len(X))


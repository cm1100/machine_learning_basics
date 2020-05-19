import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd


style.use('ggplot')


class K_Means():
    def __init__(self,k=2,tol=0.0001,max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter

    def fit(self,data):
        self.centroids={}

        for i in range(self.k):
            self.centroids[i]=data[i]

        for i in range(self.max_iter):

            self.classifications={}

            for c in range(len(self.centroids)):
                self.classifications[c]=[]
            for feature in data:
                distances=[np.linalg.norm(feature - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature)

            prev_centroids=dict(self.centroids)

            for classification in self.classifications:

                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized=True
            for c in self.centroids:
                orignal_centroid=prev_centroids[c]
                cur_centroid=self.centroids[c]
                if np.sum((cur_centroid-orignal_centroid)/orignal_centroid*100) > self.tol:
                    print(np.sum((cur_centroid-orignal_centroid)/orignal_centroid*100))
                    optimized = False

            if optimized:
                break


    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



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

clf=K_Means()
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
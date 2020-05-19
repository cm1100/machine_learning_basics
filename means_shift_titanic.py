import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

style.use('ggplot')

df = pd.read_excel('titanic.xls')
orignal_df = pd.read_excel('titanic.xls')
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
#print(df.head())

X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)
y=np.array(df['survived'])

clf=MeanShift()
clf.fit(X)

labels=clf.labels_
print(labels)
cluster_centers=clf.cluster_centers_

orignal_df['cluster_group']=np.nan

for i in range(len(X)):
    orignal_df['cluster_group'].iloc[i]=labels[i]

n_clusters_=len(np.unique(labels))

survival_rates={}
for i in range(n_clusters_):
    temp_df=orignal_df[ (orignal_df['cluster_group']==float(i)) ]
    survival_cluster=temp_df[ (temp_df['survived']==1) ]
    survival_rate =len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

print(orignal_df[ (orignal_df['cluster_group']==1) ].describe())



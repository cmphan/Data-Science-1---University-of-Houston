#Cuong Phan
#Confusion Matrix
import pandas as pd
from sklearn.metrics import confusion_matrix 
y_actu = [2,0,2,2,0,1,1,2,2,0,1,2]
y_pred = [0,0,2,1,0,2,1,0,2,0,2,2]
confusion_matrix(y_actu, y_pred)

#using pandas
y_actu = pd.Series([2,0,2,2,0,1,1,2,2,0,1,2],name='Actual')
y_pred = pd.Series([0,0,2,1,0,2,1,0,2,0,2,2],name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)

df_conf_norm = df_confusion/df_confusion.sum(axis=1)
print(df_conf_norm)

import matplotlib.pyplot as plt
import matplotlib as mpl
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) #imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks,df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)  
plot_confusion_matrix(df_confusion)
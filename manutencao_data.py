import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

data_treino = pd.read_csv('./data/desafio_manutencao_preditiva_treino.csv')
data_teste = pd.read_csv('./data/desafio_manutencao_preditiva_teste.csv')

#print(data_treino.info())

def Encoder(feature):
    le = preprocessing.LabelEncoder()
    data_teste[feature] = le.fit_transform(data_teste[feature])
         
cat_columns = ["failure_type"]
for feature in cat_columns:
    Encoder(feature)

sns.heatmap(data_treino.corr(), annot=True, square=True, cmap='YlGnBu')
plt.show()

#df_life[['Life expectancy ', 'Status']].groupby(['Status']).mean()
#sns.scatterplot(x = "Life expectancy ", y = "Income composition of resources", data = df_life)
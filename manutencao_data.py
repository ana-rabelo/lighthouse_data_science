import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sns.set_style("darkgrid")

data_treino = pd.read_csv("./data/desafio_manutencao_preditiva_treino.csv")
data_teste = pd.read_csv("./data/desafio_manutencao_preditiva_teste.csv")

data_teste.drop("product_id", axis=1, inplace=True)
data_treino.drop("product_id", axis=1, inplace=True)

rename_columns = {
    "udi": "id",
    "type": "tipo",
    "air_temperature_k": "temp_ar_kelvin",
    "process_temperature_k": "temp_processo_kelvin",
    "rotational_speed_rpm": "velocidade_rotacao_rpm",
    "torque_nm": "torque_nm",
    "tool_wear_min": "desgaste_ferramenta_min",
    "failure_type": "tipo_falha",
}

data_treino.rename(columns=rename_columns, inplace=True)
data_teste.rename(columns=rename_columns, inplace=True)


def Encoder(feature, df):
    le = preprocessing.LabelEncoder()
    df[feature] = le.fit_transform(df[feature])


cat_columns = ["tipo_falha", "tipo"]
for feature in cat_columns:
    Encoder(feature, data_treino)
for feature in cat_columns:
    Encoder("tipo", data_teste)

sns.heatmap(data_treino.corr(), annot=True, square=True, cmap="rocket")
sns.set(rc={"figure.figsize": (10, 10)})
plt.show()

X = data_treino.drop("tipo_falha", axis=1)
y = data_treino["tipo_falha"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy*100:.2f}%")

predictions = model.predict(data_teste)

output = pd.DataFrame(
    {
        "id": data_teste.id,
        "tipo": data_teste.tipo,
        "temp_ar_kelvin": data_teste.temp_ar_kelvin,
        "temp_processo_kelvin": data_teste.temp_processo_kelvin,
        "velocidade_rotacao_rpm": data_teste.velocidade_rotacao_rpm,
        "torque_nm": data_teste.torque_nm,
        "velocidade_rotacao_rpm": data_teste.velocidade_rotacao_rpm,
        "tipo_falha": predictions,
    }
)
output.to_csv("./data/predicted.csv", index=False)

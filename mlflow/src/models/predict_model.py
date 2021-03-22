import mlflow
logged_model = 'file:///mnt/d/Alura/2104-mlflowlyfecycle/codigo/mlflow/mlruns/1/d999f51f91094a00ace0648fc715400d/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv('data/processed/casas_X.csv')
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('precos.csv')
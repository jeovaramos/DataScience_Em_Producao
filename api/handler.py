import pickle
import pandas as pd
from rossmann.Rossmann import Rossmann
from flask import Flask, request, Response

# Loading model
model = pickle.load(
    open(str(
        'C:/Users/jeova/comunidade_ds/'
        'DataScience_Em_Producao/model/model_rossmann.pkl'),
        'rb')
)

# Initialize API
app = Flask(__name__)


@app.route('/rossmann/predict', methods=['POST'])
def rosmann_predict():
    test_json = request.get_json()

    if test_json:  # Check if there is data
        # Unique example
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])

        # Multiple example
        else:
            test_raw = pd.DataFrame(
                test_json, columns=test_json[0].keys())

        # Instantiate Rossmann class
        pipeline = Rossmann()

        # Data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # Feature engeneering
        df2 = pipeline.feature_engineering(df1)

        # Data preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

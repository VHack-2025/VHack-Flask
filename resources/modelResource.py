import pandas as pd
import joblib
from flask_restful import Resource
from flask import request, jsonify

rfc = joblib.load("./model/random_forest.joblib")

class modelsPOSTResource(Resource):
    def post(self):
        data = request.json
        df_json = {
            'LoanAmount': [data['LoanAmount']],
            'LoanCurrency': [data['LoanCurrency']],
            'LoanDuration': [data['LoanDuration']],
            'TotalLoanCollatoralAmount': [data['TotalLoanCollatoralAmount']],
            'LoanStartDates': [data['LoanStartDates']],
            'InstallmentDuration': [data['InstallmentDuration']],
            'BaseInterestRate': [data['BaseInterestRate']],
            'InterestRate': [data['InterestRate']]
        }
        df = pd.DataFrame(data=df_json)
        df['LoanCurrency'] = df['LoanCurrency'].apply(lambda x: 1 if x == "eth" else -1)
        df['LoanStartDates'] = pd.to_datetime(df['LoanStartDates'])
        df['LoanStartDates'] = df['LoanStartDates'].astype('int64')
        loan_pred = rfc.predict(df)
        return jsonify({"result": int(loan_pred[0])})
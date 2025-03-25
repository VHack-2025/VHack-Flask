import pandas as pd
import joblib
from flask_restful import Resource
from flask import request, jsonify

rfc = joblib.load("./model/random_forest.joblib")

class modelsPOSTResource(Resource):
    def post(self):
        data = request.json
        df_json = {
            'Age': [data['Age']],
            'EmploymentStatus': [data['EmploymentStatus']],
            'EducationLevel': [data['EducationLevel']],
            'LoanAmount': [data['LoanAmount']],
            'LoanCurrency': [data['LoanCurrency']],
            'LoanDuration': [data['LoanDuration']],
            'LoanPurpose': [data['LoanPurpose']],
            'TotalLoanCollatoralAmount': [data['TotalLoanCollatoralAmount']],
            'LoanStartDate': [data['LoanStartDate']],
            'InstallmentDuration': [data['InstallmentDuration']],
            'BaseInterestRate': [data['BaseInterestRate']],
            'InterestRate': [data['InterestRate']]
        }
        df = pd.DataFrame(data=df_json)
        df['LoanCurrency'] = df['LoanCurrency'].apply(lambda x: 1 if x == "eth" else -1)
        df['LoanStartDate'] = pd.to_datetime(df['LoanStartDate'])
        df['LoanStartDate'] = df['LoanStartDate'].astype('int64')
        
        df.loc[df['EmploymentStatus'] == "Employed", 'EmploymentStatus'] = 0
        df.loc[df['EmploymentStatus'] == "Self-Employed", 'EmploymentStatus'] = 1
        df.loc[df['EmploymentStatus'] == "Student", 'EmploymentStatus'] = 2
        df.loc[df['EmploymentStatus'] == "Unemployed", 'EmploymentStatus'] = 3
        
        df.loc[df['EducationLevel'] == "No Education", 'EducationLevel'] = 0
        df.loc[df['EducationLevel'] == "Secondary", 'EducationLevel'] = 1
        df.loc[df['EducationLevel'] == "Bachelor", 'EducationLevel'] = 2
        df.loc[df['EducationLevel'] == "Master", 'EducationLevel'] = 3
        df.loc[df['EducationLevel'] == "Doctorate", 'EducationLevel'] = 4
        
        df.loc[df['LoanPurpose'] == "Property", 'LoanPurpose'] = 0
        df.loc[df['LoanPurpose'] == "Automobile", 'LoanPurpose'] = 1
        df.loc[df['LoanPurpose'] == "Education", 'LoanPurpose'] = 2
        df.loc[df['LoanPurpose'] == "Debt Consolidation", 'LoanPurpose'] = 3
        df.loc[df['LoanPurpose'] == "Medical", 'LoanPurpose'] = 4
        df.loc[df['LoanPurpose'] == "Personal", 'LoanPurpose'] = 5
        df.loc[df['LoanPurpose'] == "Other", 'LoanPurpose'] = 6
        
        loan_pred = rfc.predict(df)
        return jsonify({"result": int(loan_pred[0])})
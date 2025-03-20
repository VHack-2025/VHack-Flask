import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of samples
num_samples = 5000

# Seed for reproducibility
np.random.seed(42)

def generate_time_based_features(num_samples):
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    return dates

application_dates = generate_time_based_features(num_samples)

data = {
    'LoanAmount': np.random.lognormal(10, 0.5, num_samples).astype(int),
    'LoanCurrency': np.random.choice(['usdc', 'eth'], num_samples,p=[0.7,0.3]),
    'LoanDuration': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120], num_samples, p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.025, 0.025]),
    'LoanPurpose': np.random.choice(['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other'], num_samples, p=[0.3, 0.2, 0.15, 0.25, 0.1]),
    'TotalLoanCollatoralAmount': np.random.lognormal(11, 1, num_samples).astype(int),
    'LoanStartDates': application_dates
}

df = pd.DataFrame(data)

df['InstallmentDuration'] = np.random.randint(1,df['LoanDuration'], num_samples)

df['BaseInterestRate'] = 0.03 + df['LoanAmount'] / 1000000 + df['LoanDuration'] / 1200
df['InterestRate'] = df['BaseInterestRate'] * (1 + np.random.normal(0, 0.1, num_samples)).clip(0.8, 1.2)

# Create a more complex loan approval rule
def loan_approval_rule(row):
    score = 0
    score += (row['LoanAmount'] - 10000) / 90000  # Loan amount factor
    score += (row['InterestRate'] - 0.05) * 10  # Interest rate factor

    if row['TotalLoanCollatoralAmount'] > 0:
        LCR = row['LoanAmount'] / row['TotalLoanCollatoralAmount']
        score -= (LCR - 0.5) * 2  # Favor lower LCR values (adjust scaling as needed)
    
    # Seasonal factor (higher approval rates in spring/summer)
    month = row['LoanStartDates'].month
    score -= 0.1 if 3 <= month <= 8 else 0
    
    # Random factor to add some unpredictability
    score += np.random.normal(0, 0.5)
    
    return 1 if score < 1 else 0  # Adjust this threshold to change overall approval rate

df['LoanApproved'] = df.apply(loan_approval_rule, axis=1)

# Save to CSV
df.to_csv('focused_synthetic_loan_data.csv', index=False)
print("\nFocused synthetic data saved to 'focused_synthetic_loan_data.csv'")

# Display final feature count
print(f"\nTotal number of features (including label): {len(df.columns)}")
print("\nFeatures:")
for column in df.columns:
    print(f"- {column}")


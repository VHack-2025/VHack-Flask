import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

# Number of samples
num_samples = 2000

# Seed for reproducibility
np.random.seed(42)

def generate_correlated_features(num_samples):
    # Generate base features
    age = np.random.normal(40, 12, num_samples).clip(18, 80).astype(int)
    education_level = np.random.choice(['No Education', 'Secondary', 'Bachelor', 'Master', 'Doctorate'], num_samples, p=[0.3, 0.2, 0.3, 0.15, 0.05])
    
    # Education affects income and credit score
    edu_impact = {'No Education': 0, 'Secondary': 0.1, 'Bachelor': 0.2, 'Master': 0.3, 'Doctorate': 0.4}
    edu_factor = np.array([edu_impact[level] for level in education_level])
    
    employment_status_probs = np.column_stack([
        0.9 - edu_factor * 0.3,  # Employed
        0.05 + edu_factor * 0.3,  # Self-Employed
        0.05 + edu_factor * 0.2,  # Student
        0.05 + edu_factor * 0.1   # Unemployed
    ])
    employment_status = np.array(['Employed', 'Self-Employed', 'Student', 'Unemployed'])[np.argmax(np.random.random(num_samples)[:, np.newaxis] < employment_status_probs.cumsum(axis=1), axis=1)]
    
    return age, education_level, employment_status

def generate_time_based_features(num_samples):
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    return dates

age, education_level, employment_status = generate_correlated_features(num_samples)
application_dates = generate_time_based_features(num_samples)

data = {
    'LoanStartDate': application_dates,
    'Age': age,
    'EmploymentStatus': employment_status,
    'EducationLevel': education_level,
    'LoanAmount': np.random.lognormal(10, 0.5, num_samples).astype(int),
    'LoanCurrency': np.random.choice(['usdc', 'eth'], num_samples,p=[0.7,0.3]),
    'LoanDuration': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120], num_samples, p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.025, 0.025]),
    'LoanPurpose': np.random.choice(['Property', 'Automobile', 'Education', 'Debt Consolidation', 'Medical', 'Personal', 'Other'], num_samples, p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]),
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
    score += (row['InterestRate'] - 0.05) * 2  # was *10, now *2

    if row['TotalLoanCollatoralAmount'] > 0:
        LCR = row['LoanAmount'] / row['TotalLoanCollatoralAmount']
        score -= (LCR - 0.5) * 2  # Favor lower LCR values (adjust scaling as needed)
    
    # Seasonal factor (higher approval rates in spring/summer)
    month = row['LoanStartDates'].month
    score -= 0.1 if 3 <= month <= 8 else 0
    
    # Age factor (slight preference for middle-aged applicants)
    score += abs(row['Age'] - 40) / 100
    
    # Education factor
    edu_score = {'No Education': 0.2, 'Secondary': 0.1, 'Bachelor': 0, 'Master': -0.1, 'Doctorate': -0.2}
    score += edu_score[row['EducationLevel']]
    
    # Random factor to add some unpredictability
    score += np.random.normal(0, 0.1)
    
    # return 1 if score < 1 else 0  # Adjust this threshold to change overall approval rate
    prob = 1 / (1 + np.exp(score))  # sigmoid-based probability
    return np.random.rand() < prob

df['LoanApproved'] = df.apply(loan_approval_rule, axis=1)

noise_strength = 0.05
interest_std_dev = df['InterestRate'].mean() * noise_strength
np.random.seed(123)

df['InterestRate'] = df['InterestRate'] + np.random.normal(0, interest_std_dev, num_samples)
df['InterestRate'] = df['InterestRate'].clip(lower=0)

df['LoanAmount'] += np.random.normal(0, df['LoanAmount'].std() * 0.1, num_samples).astype(int)
df['LoanAmount'] = df['LoanAmount'].clip(lower=1000)

df['TotalLoanCollatoralAmount'] += np.random.normal(0, df['TotalLoanCollatoralAmount'].std() * 0.1, num_samples).astype(int)
df['TotalLoanCollatoralAmount'] = df['TotalLoanCollatoralAmount'].clip(lower=0)

df['Age'] += np.random.normal(0, 3, num_samples).astype(int)
df['Age'] = df['Age'].clip(lower=18, upper=80)

flip_mask = np.random.rand(num_samples) < 0.1  # 10% corruption
edu_categories = ['No Education', 'Secondary', 'Bachelor', 'Master', 'Doctorate']
emp_categories = ['Employed', 'Self-Employed', 'Student', 'Unemployed']

df.loc[flip_mask, 'EducationLevel'] = np.random.choice(edu_categories, size=flip_mask.sum())
df.loc[flip_mask, 'EmploymentStatus'] = np.random.choice(emp_categories, size=flip_mask.sum())

# Save to CSV
df.to_csv('focused_synthetic_loan_data.csv', index=False)
print("\nFocused synthetic data saved to 'focused_synthetic_loan_data.csv'")

# Display final feature count
print(f"\nTotal number of features (including label): {len(df.columns)}")
print("\nFeatures:")
for column in df.columns:
    print(f"- {column}")
import sklearn.preprocessing as sk
import pandas as pd

filename = "data/census_income.csv"
continuous_instances = ["age", "education_num", "hours_per_week"]
instances = ["age", "education_num", "race", "sex", "hours_per_week", "greater_than_50"]
print(instances)

income = pd.read_csv(filename, usecols=instances)
income['race'] = income['race'].map({
    'White': 1,
    'Asian-Pac-Islander': 2,
    'Amer-Indian-Eskimo': 3,
    'Other': 4,
    'Black': 5
})
income['greater_than_50'] = income['greater_than_50'].map({
    '<=50K': 1,
    '>50K': 0
})
income['sex'] = income['sex'].map({
    'Male': 1,
    'Female': 0
})

for field in continuous_instances:
    income[field] = income[field] / income[field].max()

income.to_csv("data/census_income_normalized.csv", index=False, header=False)
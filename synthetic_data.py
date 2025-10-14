import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()

BASE_PARAMS = {
    'chemical': {'pH_base': 5.5, 'nitrate_base': 15.0, 'temp_base': 22.0},
    'textile': {'pH_base': 7.5, 'nitrate_base': 10.0, 'temp_base': 28.0},
    'food_processing': {'pH_base': 6.8, 'nitrate_base': 5.0, 'temp_base': 18.0}
}
POLLUTION_EFFECTS = {
    'chemical': {'pH_delta': [-2.5, 2.5], 'nitrate_mult': [3.0, 7.0]},
    'textile': {'pH_delta': [-1.0, 1.0], 'nitrate_mult': [2.0, 5.0]},
    'food_processing': {'pH_delta': [-0.5, 0.5], 'nitrate_mult': [1.5, 3.0]}
}

def generate_row(timestamp, factory_id, industry_type, pollution_flag, season):
    params = BASE_PARAMS[industry_type]

    pH = np.clip(params['pH_base'] + np.random.normal(0, 0.2), 4.0, 9.0)
    nitrate = np.clip(params['nitrate_base'] + np.random.normal(0, 1.5), 0, 80)

    temp_variation = {'winter': -5, 'spring': 0, 'summer': 5, 'autumn': 0}[season]
    temp = np.clip(params['temp_base'] + temp_variation + np.random.normal(0, 1.0), 10.0, 35.0)

    if pollution_flag == 1:
        effects = POLLUTION_EFFECTS[industry_type]
        pH += np.random.uniform(*effects['pH_delta'])
        nitrate *= np.random.uniform(*effects['nitrate_mult'])
    else:
        pH += np.random.uniform(-0.1, 0.1)
        nitrate += np.random.uniform(0.1, 1.0)

    if random.random() < 0.07: pH = np.nan
    if random.random() < 0.07: nitrate = np.nan
    if random.random() < 0.07: temp = np.nan

    pH = np.clip(pH, 3.0, 10.0)
    nitrate = np.clip(nitrate, 0.0, 100.0)
    temp = np.clip(temp, 5.0, 40.0)

    return {
        'Timestamp': timestamp,
        'Factory_ID': factory_id,
        'Industry_Type': industry_type,
        'pH': round(pH, 2),
        'Nitrate_Concentration': round(nitrate, 2),
        'Temperature': round(temp, 2),
        'Pollution_Flag': pollution_flag
    }

def generate_synthetic_dataset(num_samples=10000):
    start_date = pd.to_datetime('2023-01-01 00:00:00')
    data = []
    factory_ids = [fake.uuid4() for _ in range(10)]
    industry_types = list(BASE_PARAMS.keys())

    for i in range(num_samples):
        timestamp = start_date + timedelta(hours=i)

        month = timestamp.month
        if 3 <= month <= 5: season = 'spring'
        elif 6 <= month <= 8: season = 'summer'
        elif 9 <= month <= 11: season = 'autumn'
        else: season = 'winter'

        factory_id = random.choice(factory_ids)
        industry_type = random.choice(industry_types)

        pollution_prob = 0.15
        if industry_type == 'chemical': pollution_prob = 0.2
        pollution_flag = 1 if random.random() < pollution_prob else 0
        row = generate_row(timestamp, factory_id, industry_type, pollution_flag, season)
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index('Timestamp').sort_index()
    return df

df_synthetic = generate_synthetic_dataset()
df_synthetic.to_csv('synthetic_river_health_data.csv')
print("Synthetic dataset generated and saved to 'synthetic_river_health_data.csv'")
print(df_synthetic.head())


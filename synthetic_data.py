import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()

base_values = {
    "chemical":       {"pH": 5.5, "nitrate": 15, "temp": 22, "turbidity": 10, "do": 7,  "conductivity": 300},
    "textile":        {"pH": 7.5, "nitrate": 10, "temp": 28, "turbidity": 20, "do": 6.5, "conductivity": 500},
    "food_processing":{"pH": 6.8, "nitrate": 5,  "temp": 18, "turbidity": 5,  "do": 8,   "conductivity": 150}
}

pollution_effect = {
    "chemical":       {"pH": (-2.5, 2.5), "nitrate": (3, 7),   "turbidity": (2, 5),  "do": (-3, -1),   "conductivity": (1.5, 3)},
    "textile":        {"pH": (-1, 1),     "nitrate": (2, 5),   "turbidity": (1.5, 3),"do": (-2, -0.5), "conductivity": (1.2, 2.5)},
    "food_processing":{"pH": (-0.5, 0.5), "nitrate": (1.5, 3), "turbidity": (1.2, 2),"do": (-1, 0),    "conductivity": (1.1, 1.8)}
}

def calculate_wqi(pH, DO, nitrate, turbidity, temp):
    pH_score = max(0, 100 - abs(pH - 7) * 15)
    do_score = np.clip((DO - 4) * 25, 0, 100)
    nitrate_score = max(0, 100 - nitrate * 5)
    turbidity_score = max(0, 100 - turbidity * 4)
    score = (
        do_score * 0.40 +
        pH_score * 0.25 +
        nitrate_score * 0.20 +
        turbidity_score * 0.15
    )
    return round(np.clip(score, 0, 100), 2)

def create_record(time, factory, industry, is_polluted, season):
    vals = base_values[industry]

    pH = vals["pH"] + np.random.normal(0, 0.2)
    nitrate = vals["nitrate"] + np.random.normal(0, 1.5)
    turbidity = vals["turbidity"] + np.random.normal(0, 3)
    temp_base = vals["temp"] + {"winter": -5, "spring": 0, "summer": 5, "autumn": 0}[season]
    water_temp = temp_base + np.random.normal(0, 1)
    do_base = vals["do"] + (0.5 if season in ["winter", "spring"] else -0.5)
    DO = do_base + np.random.normal(0, 0.5)
    conductivity = vals["conductivity"] + np.random.normal(0, 50)

    if is_polluted:
        eff = pollution_effect[industry]
        pH += np.random.uniform(*eff["pH"])
        nitrate *= np.random.uniform(*eff["nitrate"])
        turbidity *= np.random.uniform(*eff["turbidity"])
        DO += np.random.uniform(*eff["do"])
        conductivity *= np.random.uniform(*eff["conductivity"])
    else:
        pH += np.random.uniform(-0.1, 0.1)
        nitrate += np.random.uniform(0.1, 1)
        turbidity += np.random.uniform(0.1, 2)
        DO += np.random.uniform(-0.1, 0.1)
        conductivity += np.random.uniform(1, 5)

    pH = np.clip(pH, 3, 10)
    nitrate = np.clip(nitrate, 0, 100)
    turbidity = np.clip(turbidity, 0, 150)
    DO = np.clip(DO, 0, 14)
    conductivity = np.clip(conductivity, 50, 2000)
    water_temp = np.clip(water_temp, 5, 40)

    if random.random() < 0.07: pH = np.nan
    if random.random() < 0.07: nitrate = np.nan
    if random.random() < 0.07: water_temp = np.nan
    if random.random() < 0.07: turbidity = np.nan
    if random.random() < 0.07: DO = np.nan
    if random.random() < 0.07: conductivity = np.nan

    if any(np.isnan([pH, DO, nitrate, turbidity, water_temp])):
        wqi = np.nan
    else:
        wqi = calculate_wqi(pH, DO, nitrate, turbidity, water_temp)

    return {
        "Timestamp": time,
        "Factory_ID": factory,
        "Industry_Type": industry,
        "pH": round(pH, 2),
        "Turbidity": round(turbidity, 2),
        "Dissolved_Oxygen": round(DO, 2),
        "Water_Temperature": round(water_temp, 2),
        "Conductivity": round(conductivity, 2),
        "Nitrate": round(nitrate, 2),
        "Water_Quality_Index": wqi,
        "Pollution_Flag": is_polluted
    }

def generate_dataset(n=10000):
    start = pd.to_datetime("2023-01-01 00:00:00")
    data = []

    factories = [fake.uuid4() for _ in range(5)]
    industries = list(base_values.keys())

    for i in range(n):
        time = start + timedelta(hours=i)

        month = time.month
        if 3 <= month <= 5:
            season = "spring"
        elif 6 <= month <= 8:
            season = "summer"
        elif 9 <= month <= 11:
            season = "autumn"
        else:
            season = "winter"

        factory = random.choice(factories)
        industry = random.choice(industries)

        prob = 0.20 if industry == "chemical" else 0.15
        polluted = 1 if random.random() < prob else 0

        entry = create_record(time, factory, industry, polluted, season)
        data.append(entry)

    df = pd.DataFrame(data)
    df = df.set_index("Timestamp").sort_index()
    return df

df = generate_dataset()
df.to_csv("synthetic_river_health_data.csv")

print("Synthetic dataset saved as synthetic_river_health_data.csv")
print(df.head())

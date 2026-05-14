import pandas as pd
import numpy as np
from src.lotr.factories import ScheduleFactory

try:
    df = pd.read_csv('results/lotr_schedule_probe_by_D/histograms_by_D.csv')
    leduc = df[(df['game'] == 'leduc_poker') & (df['D'] == 5)].copy()

    schedules = ['uniform', 'near_active', 'far_backoff', 'max_backoff']
    rho = 0.5
    D = 5

    print(f'{"Schedule":<15} | {"k":<2} | {"Empirical":<10} | {"Analytic":<10}')
    print("-" * 45)

    for s_name in schedules:
        # Empirical
        row = leduc[leduc['schedule'] == s_name]
        if row.empty:
            print(f"{s_name:<15} | No empirical data")
            continue
        
        # Sum counts across kinds (p0, p1, chance) for k=1..5
        emp_counts = []
        for k in range(1, D+1):
            col_sum = row[f"backoff_{k}_p0"].iloc[0] + row[f"backoff_{k}_p1"].iloc[0] + row[f"backoff_{k}_chance"].iloc[0]
            emp_counts.append(col_sum)
        
        total_diverged = sum(emp_counts)
        emp_        emp_        emp_        empl_        emp_      0         emp_     ts        emp_        emp_        emp_        e =    ed        emp_        emp_        emp_        empl_        emp_      0       ei        emp_        emp_        emp_        empl_        emp_      eights)
        ana_probs = [w / sum_w for w in weights]
        
        for k in range(1, D+1):
            s_label = s_name if k==1 else ""
            print(f"{s_label:<15} | {k:<2} | {emp_probs[k-1]:<10.4f} | {ana_probs[k-1]:<10.4f}")
        print("-" * 45)
except Exception as e:
    print(f"Error: {e}")

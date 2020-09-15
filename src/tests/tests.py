# import src.models.build_features as bf
import pandas as pd

if __name__ == '__main__':
    pd_data=pd.read_csv('data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_data=pd_data.sort_values('date',ascending=True).copy()

"""
Created by Sathvik Koneru on 6/15/18.
"""

import pandas as pd

# reading census csv file
census_df = pd.read_csv('census.csv')
# print(census_df.head())


# Which state has the most counties in it?
def answer_five():
    state_df = census_df[census_df['SUMLEV'] == 50]
    return state_df.groupby('STNAME').count()['SUMLEV'].idxmax()


print(answer_five())

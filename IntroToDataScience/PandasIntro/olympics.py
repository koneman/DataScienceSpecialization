"""
Created by Sathvik Koneru on 6/14/18.
"""

import pandas as pd

# read olympics.csv
df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

# rename columns
for col in df.columns:
    if col[:2] == '01':
        df.rename(columns={col: 'Gold' + col[4:]}, inplace=True)
    if col[:2] == '02':
        df.rename(columns={col: 'Silver' + col[4:]}, inplace=True)
    if col[:2] == '03':
        df.rename(columns={col: 'Bronze' + col[4:]}, inplace=True)
    if col[:1] == '№':
        df.rename(columns={col: '#' + col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(')  # split the index by '('

df.index = names_ids.str[0]  # the [0] element is the country name (new index)
df['ID'] = names_ids.str[1].str[:3]  # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
# print(df.head())


# Which country has won the most gold medals in summer games?
def answer_one():
    summer_gold = df['Gold'].idxmax()
    return summer_gold


# Which country had the biggest difference
# between their summer and winter gold medal counts?
def answer_two():
    summer_gold = df['Gold']
    winter_gold = df['Gold.1']
    max_diff = (summer_gold - winter_gold).idxmax()
    return max_diff


# Which country has the biggest difference between their summer gold medal counts
# and winter gold medal counts relative to their total gold medal count?
# (Summer Gold−Winter Gold)/Total Gold
# Only include countries that have won at least 1 gold in both summer and winter.
def answer_three():
    summer_gold = df['Gold']
    winter_gold = df['Gold.1']
    return "YOUR ANSWER HERE"


print(answer_two())

"""
Created by Sathvik Koneru on 7/30/18.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National',
          'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana',
          'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho',
          'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan',
          'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico',
          'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa',
          'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana',
          'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California',
          'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island',
          'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia',
          'ND': 'North Dakota', 'VA': 'Virginia'}

# file accessors
homes = pd.read_csv('City_Zhvi_AllHomes.csv')
gdplev = pd.read_excel('gdplev.xls', names=['quarter', 'gdp'], skiprows=219, usecols='E,G')
university_towns = pd.read_table('university_towns.txt')


def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ],
    columns=["State", "RegionName"]  )

    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''

    state_region_list = []
    state_name = ''
    region_name = ''

    with open('university_towns.txt') as file:
        for row in file.readlines():
            this_row = row[:-1]
            if this_row[-6:] == '[edit]':
                state_name = this_row[:-6]
            elif ' (' in this_row:
                region_name = this_row[:this_row.index('(') - 1]
                state_region_list.append([state_name, region_name])
            else:
                default = this_row
                state_region_list.append([state_name, default])

    state_region_df = pd.DataFrame(state_region_list, columns=['State', 'Region'])
    return state_region_df


def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''
    # A quarter is a specific three month period, Q1 is January through March, Q2 is April through June,
    # Q3 is July through September, Q4 is October through December.
    # A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two
    # consecutive quarters of GDP growth.
    for i in range(1, 65):
        if (gdplev.loc[i, "gdp"] < gdplev.loc[i-1, "gdp"]) & (gdplev.loc[i+1, "gdp"] < gdplev.loc[i, "gdp"]):
            return gdplev.loc[i, 'quarter']


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''
    for i in range(gdplev.index[gdplev['quarter'] == get_recession_start()][0],65):
        if gdplev.loc[i, "gdp"] > gdplev.loc[i-1, "gdp"] and gdplev.loc[i-1, "gdp"] > gdplev.loc[i-2, "gdp"]:
            return gdplev.loc[i, 'quarter']


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''
    # A recession bottom is the quarter within a recession which had the lowest GDP.
    recession_start = get_recession_start()
    start_index = gdplev[gdplev['Quarter'] == recession_start].index.tolist()[0]
    recession_end = get_recession_end()
    end_index = gdplev[gdplev['Quarter'] == recession_end].index.tolist()[0]
    gdplev = gdplev.iloc[start_index:end_index + 1]
    return gdplev[gdplev['GDP'] == gdplev['GDP'].min()].iloc[0]['Quarter']



print(get_recession_bottom())


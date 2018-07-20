"""
Created by Sathvik Koneru on 7/19/18.
"""

import pandas as pd
import numpy as np

# setting up energy dataframe cleanly from 'Energy Indicators.xls'
energy = pd.read_excel('Energy Indicators.xls')
energy = energy[16:243]
energy = energy.drop(energy.columns[[0, 1]], axis=1)
energy = energy.rename(index=str, columns={"Environmental Indicators: Energy": "Country",
                                           "Unnamed: 3": "Energy Supply",
                                           "Unnamed: 4": "Energy Supply per Capita",
                                           "Unnamed: 5": "% Renewable"})
energy["Energy Supply"] *= 1000  # convert energy supply column to gigajoules
energy = energy.replace("...", np.nan)

energy["Country"] = energy["Country"].replace({"Republic of Korea": "South Korea",
                                             "United States of America20": "United States",
                                             "United Kingdom of Great Britain and Northern Ireland19": "United Kingdom",
                                             "China, Hong Kong Special Administrative Region3": "Hong Kong",
                                             "Bolivia (Plurinational State of)": "Bolivia",
                                             "Australia1": "Australia","Switzerland17": "Switzerland",
                                             "Venezuela (Bolivarian Republic of)": "Venezuela",
                                             "Ukraine18": "Ukraine"})

# setting up GDP dataframe from world_bank.csv
GDP = pd.read_csv('world_bank.csv', skiprows=4)
GDP.rename(columns={"Country Name": "Country"}, inplace=True) # rename for the merge
GDP["Country"] = GDP["Country"].replace({"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran",
                                       "Hong Kong SAR, China": "Hong Kong"})
GDP.set_index('Country')

# setting up scimen datafram from scimagojr.xlsx
ScimEn = pd.read_excel('scimagojr-3.xlsx')

country_df = pd.merge(pd.merge(energy, GDP, on='Country'), ScimEn, on='Country')
country_df = country_df.set_index('Country')
country_df = country_df[['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
                         'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita',
                         '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]

# country_df = country_df.loc["Rank"]

#country_df = (country_df.loc[country_df['Rank'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])])

country_df.set_index('Rank')
country_df.sort_values('Rank', inplace=True)
country_df = country_df[country_df["Rank"] <= 15]


def answer_one():
    return country_df

# The previous question joined three datasets then reduced this to just the top 15 entries.
# When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
def answer_two():
    # subtract full merge from intersection
    full_df = pd.merge(pd.merge(energy, GDP, on='Country', how='outer'), ScimEn, on='Country', how='outer')
    intersect_df = pd.merge(pd.merge(energy, GDP, on='Country'), ScimEn, on='Country')
    return len(full_df) - len(intersect_df)

print(answer_two())

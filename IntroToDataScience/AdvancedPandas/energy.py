"""
Created by Sathvik Koneru on 7/19/18.
"""

import pandas as pd
import numpy as np


def answer_one():
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
                                                   "Australia1": "Australia", "Switzerland17": "Switzerland",
                                                   "Venezuela (Bolivarian Republic of)": "Venezuela",
                                                   "Ukraine18": "Ukraine"})

    # setting up GDP dataframe from world_bank.csv
    GDP = pd.read_csv('world_bank.csv', skiprows=4)
    GDP.rename(columns={"Country Name": "Country"}, inplace=True)  # rename for the merge
    GDP["Country"] = GDP["Country"].replace({"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran",
                                             "Hong Kong SAR, China": "Hong Kong"})
    GDP.set_index('Country')

    # setting up scimen datafram from scimagojr.xlsx
    ScimEn = pd.read_excel('scimagojr-3.xlsx')

    country_df = pd.merge(pd.merge(energy, GDP, on='Country'), ScimEn, on='Country')
    country_df = country_df.set_index('Country')
    country_df = country_df[['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
                             'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita',
                             '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
                             '2015']]

    country_df.set_index('Rank')
    country_df.sort_values('Rank', inplace=True)
    country_df = country_df[country_df["Rank"] <= 15]

    return country_df

# The previous question joined three datasets then reduced this to just the top 15 entries.
# When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
def answer_two():
    # subtract full merge from intersection
    full_df = pd.merge(pd.merge(energy, GDP, on='Country', how='outer'), ScimEn, on='Country', how='outer')
    intersect_df = pd.merge(pd.merge(energy, GDP, on='Country'), ScimEn, on='Country')
    return len(full_df) - len(intersect_df)


# average GDP over last 10 years for each country
def answer_three():
    Top15 = answer_one()
    avgGDP = (Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                     '2014', '2015']].mean(axis=1)).sort_values(ascending=False).rename('avgGDP')
    return avgGDP


# how much did GDP over 10 last years change for the country with the 6th largest GDP
def answer_four():
    Top15 = answer_one()
    avgGDP = answer_three()
    sixth_largest = avgGDP.index[5]
    GDP_change = Top15.loc[sixth_largest]['2015'] - Top15.loc[sixth_largest]['2006']
    return GDP_change


# return avg energy supply per capita
def answer_five():
    Top15 = answer_one()
    return Top15["Energy Supply per Capita"].mean()


# country with max % renewables - return tuple of name and %
def answer_six():
    Top15 = answer_one()
    renewable_percent = Top15["% Renewable"]
    max_renewable = renewable_percent.max()
    max_country = renewable_percent.sort_values(ascending=False).index[0]
    return (max_country, max_renewable)


# create new column that is the ratio of Self-Citations to Total Citations
# find maximum value for this new column, and what country has the highest ratio
# return a tuple with the name of the country and the ratio
def answer_seven():
    Top15 = answer_one()
    Top15["Self-Citations:Total Citations"] = Top15["Self-citations"]/Top15["Citations"]
    sorted_ratio = Top15["Self-Citations:Total Citations"].sort_values(ascending=False)
    max_country = sorted_ratio.index[0]
    max_country_value = sorted_ratio.iloc[0]
    return (max_country, max_country_value)


#  Create a column that estimates the population using Energy Supply and Energy Supply per capita.
#  What is the third most populous country according to this estimate?
def answer_eight():
    Top15 = answer_one()
    energy_supply = Top15["Energy Supply"]
    energyPerCapita = Top15["Energy Supply per Capita"]
    Top15["Population Ratio"] = energy_supply/energyPerCapita
    return Top15["Population Ratio"].sort_values(ascending=False).index[2]


# Create a column that estimates the number of citable documents per person. What is the correlation between the number
# of citable documents per capita and the energy supply per capita? Use the .corr() method.
def answer_nine():
    Top15 = answer_one()
    Top15["Population Est"] = Top15["Energy Supply"]/Top15["Energy Supply per Capita"]
    Top15["Citable docs per capita"] = Top15["Citable documents"]/Top15["Population Est"]
    x = Top15[["Energy Supply per Capita", "Citable docs per capita"]].corr()
    return x.iloc[0]


def plot9():
    import matplotlib as plt
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries
# in the top 15, and a 0 if the country's % Renewable value is below the median. This function should return
#  a series named HighRenew whose index is the country name sorted in ascending order of rank.
def answer_ten():
    Top15 = answer_one()
    median_renewable = Top15["% Renewable"].median()
    Top15['HighRenew'] = [1 if x >= median_renewable else 0 for x in Top15["% Renewable"]]
    Top15.sort_values(by='Rank', inplace=True)
    return Top15['HighRenew']


# Create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum,
# mean, and std deviation for the estimated population of each country
def answer_eleven():
    ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}
    Top15 = answer_one()

    Continents = pd.DataFrame(columns=['size','sum','mean','std'])
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']

    for continent, frame in Top15.groupby(ContinentDict):
        Continents.loc[continent] = [len(frame), frame['PopEst'].sum(), frame['PopEst'].mean(),frame['PopEst'].std()]
    return Continents


# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins.
def answer_twelve():
    Top15 = answer_one()
    ContinentDict  = {'China':'Asia',
                  'United States':'North America',
                  'Japan':'Asia',
                  'United Kingdom':'Europe',
                  'Russian Federation':'Europe',
                  'Canada':'North America',
                  'Germany':'Europe',
                  'India':'Asia',
                  'France':'Europe',
                  'South Korea':'Asia',
                  'Italy':'Europe',
                  'Spain':'Europe',
                  'Iran':'Asia',
                  'Australia':'Australia',
                  'Brazil':'South America'}
    Top15 = Top15.reset_index()
    Top15['Continent'] = [ContinentDict[country] for country in Top15['Country']]
    Top15['bins'] = pd.cut(Top15['% Renewable'],5)
    return Top15.groupby(['Continent','bins']).size()



# Convert the Population Estimate series to a string with thousands separator (using commas)
def answer_thirteen():
    Top15 = answer_one()
    Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).astype(float)
    return Top15['PopEst'].apply(lambda x: '{0:,}'.format(x))




print(answer_thirteen())

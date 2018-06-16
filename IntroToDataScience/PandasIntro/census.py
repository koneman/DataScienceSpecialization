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


# Only looking at the three most populous counties for each state, what are the three most populous states
# (in order of highest population to lowest population)? Use CENSUS2010POP.
def answer_six():
    pop_df = census_df[census_df['SUMLEV'] == 50]
    ordered_counties_df = pop_df.sort_values(by=['STNAME', 'CENSUS2010POP'], ascending=False).groupby('STNAME').head(3)
    highest3 = ordered_counties_df.groupby('STNAME').sum().sort_values(by='CENSUS2010POP').head(3).index.tolist()
    return highest3


# Which county has had the largest absolute change in population within the period 2010-2015?
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then
# its largest change in the period would be |130-80| = 50.
def answer_seven():
    pop_df = census_df[['STNAME', 'CTYNAME', 'POPESTIMATE2015', 'POPESTIMATE2014', 'POPESTIMATE2013', 'POPESTIMATE2012'
        , 'POPESTIMATE2011', 'POPESTIMATE2010']]
    pop_df = pop_df[pop_df['STNAME'] != pop_df['CTYNAME']] # remove overlap
    index = (pop_df.max(axis=1) - pop_df.min(axis=1)).idxmax() # return index where max-min occurs
    return census_df.loc[index]['CTYNAME']


# In this datafile, the United States is broken up into four regions using the "REGION" column.
# Create a query that finds the counties that belong to regions 1 or 2, whose name starts
# with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
# This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME']
# and the same index ID as the census_df (sorted ascending by index).
def answer_eight():
    counties_df = census_df[census_df['SUMLEV'] == 50]
    custom_df = counties_df[((counties_df['REGION'] == 1) | (counties_df['REGION'] == 2)) &
        (counties_df['CTYNAME'] == 'Washington County') & (counties_df['POPESTIMATE2015'] >
        counties_df['POPESTIMATE2014'])][['STNAME', 'CTYNAME']]
    return custom_df


print(answer_eight())

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import ast

from lib.functions_data import *

# +
import sys
from pathlib import Path
import os
cwd = os.getcwd()
parent = str(Path(cwd).parents[0])
sys.path.append(parent)

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# +
#Only the columns we need for the analysis to save memory
cols = ['eudract_number_with_country', 'date_of_competent_authority_decision',
       'clinical_trial_type', 'national_competent_authority', 'eudract_number',
       'date_on_which_this_record_was_first_entered_in_the_eudract_data',
       'trial_status', 'date_of_the_global_end_of_the_trial', 'trial_results']

#You can copy this URL if you want to download the full raw data
data_link = 'https://www.dropbox.com/s/4qt0msiipyn7crm/euctr_euctr_dump-2020-12-03-095517.csv.zip?dl=1'

dec = pd.read_csv(data_link, compression='zip', low_memory=False, usecols=cols)

#This is additional data from the results page we need for certain analysese
results_info = pd.read_csv(parent + '/data/euctr_data_quality_results_scrape_dec_2020.csv')
results_info['trial_start_date'] = pd.to_datetime(results_info.trial_start_date)
# -

#Quick look at the spread of trial statuses on the EUCTR
dec.trial_status.value_counts(dropna=False)

# The "date_of_competent_authority_decision" field has 2 nonsensical year values in which the correct value can reasonably be derived from context. We fix those below:
#
# https://www.clinicaltrialsregister.eu/ctr-search/trial/2009-016759-22/DK
#
# https://www.clinicaltrialsregister.eu/ctr-search/trial/2006-006947-30/FR

# +
ind = dec[dec.date_of_competent_authority_decision.notnull() & 
          dec.date_of_competent_authority_decision.str.contains('210')].index
ind = ind.to_list()[0]

ind_2 = dec[dec.date_of_competent_authority_decision.notnull() & 
            dec.date_of_competent_authority_decision.str.contains('2077')].index
ind_2 = ind_2.to_list()[0]

dec.at[ind, 'date_of_competent_authority_decision'] = '2010-06-18'
dec.at[ind_2, 'date_of_competent_authority_decision'] = '2007-04-05'

# +
#get rid of all protocols from non EU/EEA countries
dec_filt = dec[dec.clinical_trial_type != 'Outside EU/EEA'].reset_index(drop=True)

#lets see how many that is:
print(len(dec) - len(dec_filt))

# +
dec_ctas = dec[['eudract_number', 'eudract_number_with_country']].groupby('eudract_number').count()['eudract_number_with_country']

print(f'There are {len(dec_ctas)} registered trials and {dec_ctas.sum()} CTAs')

# +
decf_ctas = dec_filt[['eudract_number', 'eudract_number_with_country']].groupby('eudract_number').count()['eudract_number_with_country']

print(f'There are {len(decf_ctas)} registered trials and {decf_ctas.sum()} CTAs')

# +
#Making dates into dates and adding a column of just the "Year" for relevant dates

dec_filt['date_on_which_this_record_was_first_entered_in_the_eudract_data'] = pd.to_datetime(dec_filt['date_on_which_this_record_was_first_entered_in_the_eudract_data'])
dec_filt['entered_year'] = dec_filt['date_on_which_this_record_was_first_entered_in_the_eudract_data'].dt.year

dec_filt['date_of_competent_authority_decision'] = pd.to_datetime(dec_filt['date_of_competent_authority_decision'])
dec_filt['approved_year'] = dec_filt['date_of_competent_authority_decision'].dt.year

# +
#You can reproduce the data on the earliest registered protocol for each country by running this cell
#with the appropriate country abbreviation. For example, 'IT' for Italy:

print(earliest_record_check(dec_filt, 'IT'))

#Uncomment this to get the date for all countries at once
#for abrev in country_abrevs.keys():
#    print(f'Country: {abrev}\nEarliest record date: {earliest_record_check(dec_filt, abrev)}')

# +
#Creating a copy of the original dataset we can mess with and
#renaming columns to better variable names

analysis_df = dec_filt.copy()
analysis_df.columns = ['eudract_number_country', 
                       'approved_date', 
                       'clinical_trial_type', 
                       'nca', 
                       'eudract_number', 
                       'date_entered', 
                       'trial_status', 
                       'completion_date', 
                       'trial_results', 
                       'entered_year', 
                       'approved_year']

#And update the NCA names to the more accurate recent names

analysis_df['nca'] = analysis_df['nca'].replace(nca_name_mapping)
# -

#Table 1
analysis_df[['nca', 'eudract_number_country']].groupby('nca').count()

# +
#You can reproduce the data on the earliest registered protocol for each country by running this cell
#with the appropriate country abbreviation. For example, 'IT' for Italy:

print(earliest_record_check(analysis_df, 'Italy - AIFA'))

#Uncomment this to get the date for all countries at once
#for abrev in country_abrevs.keys():
#    print(f'Country: {abrev}\nEarliest record date: {earliest_record_check(dec_filt, abrev)}')
# -

#lastly this is helpful to have the country names in order
ordered_countries_original = list(dec_filt.national_competent_authority.value_counts().index)
ordered_countries_new = list(analysis_df.nca.value_counts().index)

# # Registrations Over Time

reg_df = analysis_df[['eudract_number', 'nca', 'date_entered', 'entered_year', 'approved_date', 'approved_year']].reset_index(drop=True)
reg_df.head()

# +
#Data for Overall Trend in Registrations

grouped_overall = reg_df[['eudract_number']].groupby([reg_df.entered_year]).count()
earliest_entered = reg_df[['eudract_number', 'date_entered']].groupby('eudract_number', as_index=False).min()
earliest_entered['year'] = earliest_entered.date_entered.dt.year
unique_trials = earliest_entered[['eudract_number', 'year']].groupby('year').count()

# +
fig, ax = plt.subplots(figsize = (12,6), dpi=300)

grouped_overall[(grouped_overall.index > 2004) & (grouped_overall.index < 2020)].plot(ax=ax, legend=False, lw=2, 
                                                                                      marker='.', markersize=12)
unique_trials[(unique_trials.index > 2004) & (unique_trials.index < 2020)].plot(ax=ax, legend=False, grid=True, 
                                                                                lw=2, marker='.', markersize=12)

ax.legend(['Total CTAs', 'Unique Trials'], bbox_to_anchor = (1, 1))
ax.set_xticks(range(2005, 2020))
ax.set_yticks(range(0,7500, 500))
plt.xlabel('CTA Entry Year', labelpad=10)
plt.ylabel('Records Entered')
plt.title('Trend in new CTA and Trial Registration on the EUCTR', pad=10)

fig.show()
# -

# Now we're interested in breaking the data down a bit further. Here we will break it down into quarters for more detailed analysis. We graph the years for which we have full EUCTR data (2005-2019).

# +
grouped = reg_df[['eudract_number']].groupby([reg_df.nca, pd.PeriodIndex(reg_df.date_entered, freq='Q')]).count()

get_index = reg_df[['eudract_number']].groupby(pd.PeriodIndex(reg_df.date_entered, freq='Q')).count()
quarters = list(get_index.index)
# -

grouped.loc['Austria - BASG'].index[0]

# +
grouped_2 = reg_df[['eudract_number']].groupby([reg_df.nca, pd.PeriodIndex(reg_df.date_entered, freq='Y')]).count()

get_index = reg_df[['eudract_number']].groupby(pd.PeriodIndex(reg_df.date_entered, freq='Y')).count()
years = list(get_index.index)
# -

grouped_year = reg_df[['eudract_number']].groupby([reg_df.nca, reg_df.entered_year]).count()
grouped_year_2 = reg_df[['eudract_number']].groupby([reg_df.nca, reg_df.approved_year]).count()

# +
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=300)
fig.suptitle("Cumulative trial registrations by NCA", y=1.05, fontsize=20)
fig.tight_layout()

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    country = grouped.loc[ordered_countries_new[x]]
    first_reporting_quarter = country[country.eudract_number > 0].index.min()
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], quarters) 
    data = pd.DataFrame({'eudract_number': adjusted_data})
    x_ticks = data.index
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_quarter, np.nan, data.eudract_number)
    
    consolidated = data[(data.index.year > 2004) & (data.index.year < 2020) & data.eudract_number.notnull()]
    
    leading_zero_check = True
    i=0
    
    while leading_zero_check:
        if consolidated.eudract_number[i] == 0:
            consolidated.at[consolidated.index[i], 'eudract_number'] = np.nan
            i+=1
        else:
            leading_zero_check = False
    
    consolidated = consolidated[consolidated.eudract_number.notnull()]
    
    cumulative = consolidated.cumsum()
    
    # Plotting the country trend
    cumulative.plot(ax=y, lw=4, sharex='col',legend=False, title=ordered_countries_new[x])
    
    #Plotting the reference line
    cumulative.loc[[cumulative.index[0], cumulative.index[-1]]].plot(ax=y, legend=False, lw=2, style='--')
    
    #int_ordinal = np.arange(0,len(consolidated.index))
    
    #z = np.polyfit(int_ordinal, consolidated.eudract_number, 1)
    #p = np.poly1d(z)
    
    #consolidated['trend'] = p(int_ordinal)
    
    #consolidated.trend.plot(ax=y)
    
    
    #if ordered_countries_original[x] == 'Slovenia - JAZMP':
    #    y.set_yticks(range(0,16,3))
    
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('Protocol Record Entry Year', labelpad=10)
    y.set_xlim(x_ticks[0], x_ticks[-1])
    
pd.set_option('mode.chained_assignment', 'warn')

plt.legend(['Cumulative Count of New CTA Registrations', 'Stable Trend Line'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.3), fontsize=15)
plt.show()
# -

# For comparison here are the raw trends

# +
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=300)
fig.suptitle("Trends in trial registrations by NCA", y=1.05, fontsize=20)
fig.tight_layout()

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    country = grouped.loc[ordered_countries_new[x]]
    first_reporting_quarter = country[country.eudract_number > 0].index.min()
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], quarters) 
    data = pd.DataFrame({'eudract_number': adjusted_data})
    x_ticks = data.index
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_quarter, np.nan, data.eudract_number)
    
    consolidated = data[(data.index.year > 2004) & (data.index.year < 2020) & data.eudract_number.notnull()]
    
    leading_zero_check = True
    i=0
    
    while leading_zero_check:
        if consolidated.eudract_number[i] == 0:
            consolidated.at[consolidated.index[i], 'eudract_number'] = np.nan
            i+=1
        else:
            leading_zero_check = False
    
    consolidated = consolidated[consolidated.eudract_number.notnull()]
    
    #if ordered_countries_new[x] != 'Croatia - MIZ':
    #    p = pv_df.loc[ordered_countries_new[x]][0]
    #    if p < .05:
    #        title = ordered_countries_new[x] + '*'
    #    else:
    #        title = ordered_countries_new[x]
    #else:
    #    title = 'Croatia - MIZ' + '^'
    
    consolidated.plot(ax=y, lw=2, sharex='col',legend=False, title=ordered_countries_new[x])
    
    #int_ordinal = np.arange(0,len(consolidated.index))
    
    #z = np.polyfit(int_ordinal, consolidated.eudract_number, 1)
    #p = np.poly1d(z)
    
    #consolidated['trend'] = p(int_ordinal)
    
    #consolidated.trend.plot(ax=y)
    
    
    if ordered_countries_original[x] == 'Slovenia - JAZMP':
        y.set_yticks(range(0,16,3))
    
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('Protocol Record Entry Year', labelpad=10)
    y.set_xlim(x_ticks[0], x_ticks[-1])
    
pd.set_option('mode.chained_assignment', 'warn')
plt.show()
# -

# Lasty, we can sense check that these dates make sense by comparing the year the CTA was entered to the date the NCA gave approval. When we graph them on top of each other, we can see that the overall trend align very well though with approvals being slightly less susceptable to large jumps.

grouped_year = reg_df[['eudract_number']].groupby([reg_df.nca, reg_df.entered_year]).count()
grouped_year_2 = reg_df[['eudract_number']].groupby([reg_df.nca, reg_df.approved_year]).count()

# +
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=300)
fig.suptitle("Trends in trial registrations by NCA", y=1.05, fontsize=20)
fig.tight_layout()

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    country = grouped_year.loc[ordered_countries_new[x]]
    first_reporting_quarter = country[country.eudract_number > 0].index.min()
    
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], range(2004, 2020))
    
    data = pd.DataFrame({'eudract_number': adjusted_data})
    x_ticks = data.index
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_quarter, np.nan, data.eudract_number)
    
    consolidated = data[(data.index > 2004) & (data.index < 2020) & data.eudract_number.notnull()]
    
    leading_zero_check = True
    i=0
    
    while leading_zero_check:
        if consolidated.eudract_number.values[i] == 0:
            consolidated.at[consolidated.index[i], 'eudract_number'] = np.nan
            i+=1
        else:
            leading_zero_check = False
    
    
    consolidated = consolidated[consolidated.eudract_number.notnull()]
    
    consolidated.plot(ax=y, lw=2, sharex='col',legend=False, title=ordered_countries_new[x])
    
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('Protocol Record Entry Year', labelpad=10)
    y.set_xlim(x_ticks[0], x_ticks[-1])
    
pd.set_option('mode.chained_assignment', 'warn')
plt.legend(['First Entered Date', 'NCA Approval Date'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.3), fontsize=15)
plt.show()

# +
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=300)
fig.suptitle("Trends in trial registrations by NCA", y=1.05, fontsize=20)
fig.tight_layout()

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    country = grouped_year.loc[ordered_countries_new[x]]
    country_2 = grouped_year_2.loc[ordered_countries_new[x]]
    first_reporting_quarter = country[country.eudract_number > 0].index.min()
    
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], range(2004, 2020))
    adjusted_data_2 = zero_out_dict(country_2.to_dict()['eudract_number'], range(2004, 2020))
    
    data = pd.DataFrame({'eudract_number': adjusted_data})
    data_2 = pd.DataFrame({'eudract_number': adjusted_data_2})
    x_ticks = data.index
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_quarter, np.nan, data.eudract_number)
    data_2['eudract_number'] = np.where(data_2.index < first_reporting_quarter, np.nan, data_2.eudract_number)
    
    consolidated = data[(data.index > 2004) & (data.index < 2020) & data.eudract_number.notnull()]
    consolidated_2 = data_2[(data_2.index > 2004) & (data_2.index < 2020) & data_2.eudract_number.notnull()]
    
    leading_zero_check = True
    i=0
    
    while leading_zero_check:
        if consolidated.eudract_number.values[i] == 0:
            consolidated.at[consolidated.index[i], 'eudract_number'] = np.nan
            i+=1
        else:
            leading_zero_check = False
    
    while leading_zero_check:
        if consolidated_2.eudract_number.values[i] == 0:
            consolidated_2.at[consolidated_2.index[i], 'eudract_number'] = np.nan
            i+=1
        else:
            leading_zero_check = False
    
    consolidated = consolidated[consolidated.eudract_number.notnull()]
    consolidated_2 = consolidated_2[consolidated_2.eudract_number.notnull()]
    
    consolidated.plot(ax=y, lw=2, sharex='col',legend=False, title=ordered_countries_new[x])
    consolidated_2.plot(ax=y, lw=2, sharex='col',legend=False)
    
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('Protocol Record Entry Year', labelpad=10)
    y.set_xlim(x_ticks[0], x_ticks[-1])
    
pd.set_option('mode.chained_assignment', 'warn')
plt.legend(['First Entered Date', 'NCA Approval Date'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.3), fontsize=15)
plt.show()
# -

# # Trial Status By Country Over Time

# +
status_df = analysis_df[['eudract_number', 'nca', 'entered_year', 'trial_status']].reset_index(drop=True)
status_df['trial_status'] = status_df.trial_status.fillna('Missing')

status_group = status_df.groupby(['nca', 'entered_year', 'trial_status'], as_index=False).count()

# +
ordered_countries = list(status_group[['nca', 'eudract_number']].groupby('nca').sum().sort_values(by='eudract_number', ascending=False).index)

#Removing these for low number of trials
ordered_countries.remove('Malta - ADM')
ordered_countries.remove('Luxembourg - Ministry of Health')
ordered_countries.remove('Cyprus - MoH-Ph.S')
# -

# Here we create our trial status categories

# +
country_status = {}

for c in status_group.nca.unique():
    country_dict = {}
    country = status_group[status_group.nca == c]
    
    completed = country[country.trial_status.isin(['Completed', 'Prematurely Ended'])][['entered_year', 'eudract_number']].groupby('entered_year').sum()
    comp_dict = completed.to_dict()['eudract_number']
    country_dict['completed'] = zero_out_dict(comp_dict, range(2004,2021))
    
    ongoing = country[country.trial_status.isin(['Ongoing', 'Restarted'])][['entered_year', 'eudract_number']].groupby('entered_year').sum()
    ong_dict = ongoing.to_dict()['eudract_number']
    country_dict['ongoing'] = zero_out_dict(ong_dict, range(2004,2021))
    
    missing = country[country.trial_status == 'Missing']
    missing_dict = pd.Series(missing.eudract_number.values, index=missing.entered_year).to_dict()
    country_dict['missing'] = zero_out_dict(missing_dict, range(2004,2021))
    
    other = country[~country.trial_status.isin(['Completed', 'Ongoing', 'Restarted', 'Prematurely Ended', 'Missing'])][['entered_year', 'eudract_number']].groupby('entered_year').sum()
    other_dict = other.to_dict()['eudract_number']
    country_dict['other'] = zero_out_dict(other_dict, range(2004,2021))
    
    country_status[c] = country_dict

# +
#Shaping up the final data so it's easy to use

regrouped = pd.DataFrame.from_dict(country_status, orient='index').stack().to_frame()[0].apply(pd.Series).reindex(
    ['completed', 'ongoing', 'other', 'missing'], level=1)

# +
#A glance at the overall trend

grouped_total = regrouped.droplevel(level=0).groupby(regrouped.droplevel(level=0).index).sum()

title='Trial Status of Protocols by Record Entry Date'


fig, ax = plt.subplots(figsize = (10,5))
grouped_total.reindex(['completed', 'ongoing', 'missing', 'other']).T.plot.bar(stacked=True, width=.9, ax=ax, 
                                                                               rot=45, title = title, legend=False)

ax.set_axisbelow(True)
ax.grid(axis='y', zorder=0)
ax.set_xlabel('Protocol Record Entry Year', labelpad=10)

plt.legend(['Completed', 'Ongoing', 'Other', 'Missing'], 
           loc='upper center', ncol=5, bbox_to_anchor = (0.5, -0.2), fontsize=12)
plt.show()

# +
overall_prct_dict = {}

for x in ordered_countries:
    g = regrouped.loc[[x]].droplevel(level=0).T
    num = g.completed.sum()
    denom = num + g.ongoing.sum() + g.missing.sum() + g.other.sum()
    overall_prct_dict[x] = num / denom

rankings_completed = pd.Series(overall_prct_dict).sort_values(ascending=False)
rankings_completed

# +
#And now a look at the trend for each EU/EEA country

fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=300)
fig.suptitle("Trial Status of Protocols by NCA", y=1.02, fontsize=20)
fig.tight_layout()
for x, y in enumerate(fig.axes):
    regrouped.loc[[rankings_completed.index[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.9, legend=False,
                                                                        sharex='col', rot=45,
                                                                        title=rankings_completed.index[x])
    
    y.set_axisbelow(True)
    y.grid(axis='y', zorder=0)
    y.set_xlabel('Protocol Record Entry Year', labelpad=10)


plt.legend(['Completed', 'Ongoing', 'Other', 'Missing'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.5), fontsize=15)

plt.show()
# -

# # Missing Completion Dates

date_df = analysis_df[['eudract_number', 'nca', 'entered_year', 'trial_status', 'completion_date', 'trial_results']].reset_index(drop=True)
date_df['trial_status'] = date_df.trial_status.fillna('Missing')
date_df['has_completion_date'] = np.where(date_df.completion_date.isna(), 0, 1)

only_completed = date_df[date_df.trial_status.isin(['Completed', 'Prematurely Ended'])].reset_index(drop=True)

# +
total_completed = only_completed[['nca', 
                                  'entered_year', 
                                  'has_completion_date']].groupby(['nca', 
                                                                   'entered_year']).count().rename({'has_completion_date': 'denominator'}, axis=1)

total_completed_date = only_completed[['nca', 'entered_year', 'has_completion_date']].groupby(['nca', 'entered_year']).sum().rename({'has_completion_date': 'numerator'}, axis=1)


# -

merged_dates = total_completed.join(total_completed_date)
merged_dates['missing_dates'] = merged_dates.denominator - merged_dates.numerator

stacked_dates = merged_dates.drop('denominator', axis=1).stack().unstack(1)

# +
overall_dates = stacked_dates.droplevel(level=0).groupby(stacked_dates.droplevel(level=0).index).sum()

title='Availability of Completion Dates for Completed Trials'

fig, ax = plt.subplots(figsize = (10,5))
overall_dates.reindex(['numerator', 'missing_dates']).T.plot.bar(stacked=True, width=.9, ax=ax, legend=False, 
                                                                 rot=45, title=title)

ax.set_axisbelow(True)
ax.grid(axis='y', zorder=0)
ax.set_xlabel('Protocol Record Entry Year', labelpad=10)

plt.legend(['Has Date', 'Missing Date'], 
           loc='upper right', fontsize=12)
plt.show()

# +
overall_comp_dict = {}

for x in ordered_countries:
    d = stacked_dates.loc[x].T
    num = d.numerator.sum()
    denom = num + d.missing_dates.sum()
    overall_comp_dict[x] = num / denom

rankings_compdate = pd.Series(overall_comp_dict).sort_values(ascending=False)
rankings_compdate

# +
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4)
fig.suptitle("Available Completion Dates for Completed Trials by NCA", y=1.05, fontsize=20)
fig.tight_layout()

for x, y in enumerate(fig.axes):
    stacked_dates.loc[[rankings_compdate.index[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.9, 
                                                                            legend=False, sharex='col', rot=45, 
                                                                            title=rankings_compdate.index[x])
    y.set_axisbelow(True)
    y.grid(axis='y', zorder=0)
    y.set_xlabel('Protocol Record Entry Year', labelpad=10)
    
plt.legend(['Has Date', 'Missing Date'], 
           loc='lower center', ncol=5, bbox_to_anchor = (-1.25, -.75), fontsize=15)

plt.show()
# -

# # Cross-checking countries listed in results with public CTAs

results_info_filt = results_info[results_info.recruitment_countries.notnull()].reset_index(drop=True)

# +
protocols = results_info_filt.trial_countries.to_list()
results_countries = results_info_filt.recruitment_countries.to_list()
start_date = results_info_filt.trial_start_date.to_list()
trial_ids = results_info_filt.trial_id.to_list()

zipped_cats = zip(trial_ids, protocols, results_countries, start_date)

results_list = compare_enrollment_registration(zipped_cats)

missing_protocols = pd.DataFrame(results_list)
missing_protocols['total_missing'] = missing_protocols.unaccounted.apply(len)
# -

acct = missing_protocols.accounted.to_list()
unacct = missing_protocols.unaccounted.to_list()

accounted_count = {}
unaccounted_count = {}
for ac, un in zip(acct, unacct):
    if ac:
        for a in ac:
            accounted_count[a] = accounted_count.get(a, 0) + 1
    if un:
        for u in un:
            unaccounted_count[u] = unaccounted_count.get(u, 0) + 1

# +
accounted_series = pd.Series(accounted_count)
unaccounted_series = pd.Series(unaccounted_count)

reg_check_no_buffer = accounted_series.to_frame().join(unaccounted_series.to_frame(), how='outer', rsuffix='unac').rename({'0': 'accounted', '0unac': 'unaccounted'}, axis=1).fillna(0)

# +
reg_check_no_buffer['total'] = reg_check_no_buffer['accounted'] + reg_check_no_buffer['unaccounted']

reg_check_no_buffer['acct_prct'] = round((reg_check_no_buffer['accounted'] / reg_check_no_buffer['total']) * 100, 2)

reg_check_no_buffer['unacct_prct'] = round((reg_check_no_buffer['unaccounted'] / reg_check_no_buffer['total']) * 100, 2)

reg_check_no_buffer.head()

# +
fig, ax = plt.subplots(figsize = (20,10))

title = 'Protocol Availability for Reported Trials By Country'

sorted_countries = reg_check_no_buffer.sort_values(by='total')
sorted_countries[['accounted', 'unaccounted']].plot.bar(stacked=True, ax=ax, title=title,
                                                        legend=False, width=.5)

ax.set_axisbelow(True)
ax.grid(axis='y', zorder=0)

rects = ax.patches[0:30]

for rect, label, y_off in zip(rects, sorted_countries.acct_prct.values, sorted_countries.total.values):
    ax.text(rect.get_x() + rect.get_width() / 2, y_off + 25, str(label) + '%', 
            ha='center', va='bottom', fontsize=10)

ax.legend(['Protocol Available', 'Protocol Unavailable'], 
           loc='upper left', fontsize=9)
    
plt.show()

# +
min_start_date = analysis_df[['eudract_number', 'entered_year']].groupby('eudract_number', as_index=False).min()

by_year_df = missing_protocols.merge(min_start_date, how='left', left_on='trial_id', right_on='eudract_number').drop('eudract_number', axis=1)

# +
fig, ax = plt.subplots(figsize = (12,6), dpi=300)

prct_missing = grouped_overall.join(to_graph)
prct_missing['missing_cta_prct'] = (prct_missing.total_missing / prct_missing.eudract_number) * 100

ax.set_ylim(0,10)
l = plt.plot(prct_missing.missing_cta_prct)

to_graph = by_year_df[['entered_year', 'total_missing']].groupby('entered_year').sum()
to_graph.index = to_graph.index.astype(int)

ax2 = plt.twinx()
ax2.set_axisbelow(True)

to_graph.plot.bar(ax=ax2, legend=False, width=.9, rot=45)

ax.grid(axis='y', zorder=0)
plt.title('Missing Protocols By Trial Entered Year', pad=10)
plt.ylabel('Missing Protocols')
    
plt.show()

# +
fig, ax = plt.subplots(figsize=(24,12), dpi = 300)

labels = [str(x) for x in range(2004,2021)]

l1 = plt.errorbar(prct_missing.index,prct_missing.missing_cta_prct, lw=5, color='orange')

plt.tick_params(axis='both', which='major', labelsize=15)
#plt.ylabel('# of Missing Trials', fontsize=25, labelpad=10)
plt.xticks(rotation=25)
plt.title("Missing CTAs by Trial Entry Year", pad = 20, fontsize = 25)

ax.set_ylim([0,10])
ax.set_ylabel('# of Missing Trials', fontsize=20, labelpad=50)

ax2 = plt.twinx()
ax2.set_axisbelow(True)
ax.yaxis.grid(linestyle='--', linewidth=.5, zorder=ax.get_zorder()-10)
ax2.bar(to_graph.index, to_graph.total_missing, tick_label=labels)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_ylabel('% Missing', fontsize=20, labelpad=30)

ax.yaxis.tick_right()
ax2.yaxis.tick_left()

ax.set_zorder(ax2.get_zorder()+1)
ax.patch.set_visible(False)

plt.legend([l1], ['% Missing'] ,fontsize=20)
plt.show()

# +
fig, ax = plt.subplots(figsize=(24,12), dpi = 300)

labels = [str(x) for x in range(2004,2021)]

l1 = plt.errorbar(prct_missing.index,prct_missing.missing_cta_prct, lw=5, color='orange')

plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel('Percent Missing', fontsize=25, labelpad=10)
plt.xticks(rotation=25)
#plt.title("Reporting Percentage and Overdue Trials By Month", pad = 20, fontsize = 30)

#plt.legend((l1), ('% missing'), loc=2, fontsize=18)
ax.set_ylim([0,10])

ax2 = plt.twinx()
ax2.set_axisbelow(True)
ax.yaxis.grid(linestyle='--', linewidth=.5, zorder=ax.get_zorder()-10)
ax2.bar(to_graph.index, to_graph.total_missing, tick_label=labels)
ax2.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel('# of Missing Trials', fontsize=25, labelpad=10)

ax.set_zorder(ax2.get_zorder()+1)
ax.patch.set_visible(False)
# -

# **Using longer lags between the first available protocol for a country to see if it makes a difference. This can be adjusted using the `offseet` parameter of the `compare_enrollment_registration` function**

# +
protocols = results_info_filt.trial_countries.to_list()
results_countries = results_info_filt.recruitment_countries.to_list()
start_date = results_info_filt.trial_start_date.to_list()
trial_ids = results_info_filt.trial_id.to_list()

zipped_cats = zip(trial_ids, protocols, results_countries, start_date)

results_sens = compare_enrollment_registration(zipped_cats, offset=6)

missing_sensitivity = pd.DataFrame(results_sens)

# +
acct_sens = missing_sensitivity.accounted.to_list()
unacct_sens = missing_sensitivity.unaccounted.to_list()

accounted_count_sens = {}
unaccounted_count_sens = {}
for ac, un in zip(acct_sens, unacct_sens):
    if ac:
        for a in ac:
            accounted_count_sens[a] = accounted_count_sens.get(a, 0) + 1
    if un:
        for u in un:
            unaccounted_count_sens[u] = unaccounted_count_sens.get(u, 0) + 1
            
accounted_series_sens = pd.Series(accounted_count_sens)
unaccounted_series_sens = pd.Series(unaccounted_count_sens)

# +
reg_check_buffer = accounted_series_sens.to_frame().join(unaccounted_series_sens.to_frame(), how='outer', rsuffix='unac').rename({'0': 'accounted', '0unac': 'unaccounted'}, axis=1).fillna(0)

reg_check_buffer['total'] = reg_check_buffer['accounted'] + reg_check_buffer['unaccounted']
reg_check_buffer['acct_prct'] = round((reg_check_buffer['accounted'] / reg_check_buffer['total']) * 100, 2)

reg_check_buffer.head()
# -

# # Trend in Results Availability by registration year

# +
reporting_by_country = analysis_df[['eudract_number', 'nca', 'entered_year', 
                                    'approved_year', 'trial_results']].reset_index(drop=True)

reporting_by_country['results_dummy'] = np.where(reporting_by_country.trial_results == 'View results', 1, 0)

# +
eu_protocol_count = reporting_by_country.groupby('eudract_number').count()[['nca']].reset_index()

eu_protocol_count.columns = ['eudract_number', 'nca_count']
# -

# Creating data for trials with only a single CTA

# +
solo_merge = reporting_by_country.merge(eu_protocol_count, how='left', on='eudract_number')

total = solo_merge[solo_merge.nca_count == 1][['nca', 'entered_year', 'results_dummy']].groupby(['nca', 'entered_year']).count().rename({'results_dummy': 'denominator'}, axis=1)

reported = solo_merge[solo_merge.nca_count == 1][['nca', 'entered_year', 'results_dummy']].groupby(['nca', 'entered_year']).sum().rename({'results_dummy': 'numerator'}, axis=1)

merged = total.join(reported)
merged['unreported'] = merged.denominator - merged.numerator

stacked = merged.drop('denominator', axis=1).stack().unstack(1)
# -

# Creating data for trials with multiple CTAs

# +
multi_set = solo_merge[solo_merge.nca_count > 1][['eudract_number', 'results_dummy', 'entered_year']].drop_duplicates()

multi_group = multi_set.groupby('eudract_number', as_index=False).agg('min')[['entered_year', 'results_dummy']].groupby('entered_year').agg({'results_dummy':['count', 'sum']})['results_dummy']
multi_group['prct'] = round((multi_group['sum'] / multi_group['count']) * 100, 2)

# +
#Graphing the overall trend for single vs multiple CTA trials

num = stacked.loc[pd.IndexSlice[:, 'numerator'], :].sum()
denom = (num + stacked.loc[pd.IndexSlice[:, 'unreported'], :].sum())

fig, ax = plt.subplots(figsize = (10,5), dpi=300)

plt.plot(range(2004,2021), multi_group.prct, marker='.', markersize=10)
plt.plot(range(2004,2021), ((num) / (denom)) * 100, marker='.', markersize=10)

ax.set_xticks(range(2004,2021))

ax.legend(['Multi-CTA', 'Single-CTA'], loc='upper right', fontsize=10)
ax.set_axisbelow(True)
ax.grid(zorder=0)

plt.ylabel('Percent Reported')
plt.xlabel('Earliest Record Entry Year', labelpad=10)

plt.title('Results Availability by Year', pad=10)
plt.show()

# +
#Single CTAs for all NCAs...could turn this into lines as well potentially

fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4)
fig.suptitle("Proportion of Single-CTA Trials Reported by Year", y=1.05, fontsize=20)
fig.tight_layout()
for x, y in enumerate(fig.axes):
    stacked.loc[[ordered_countries_new[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.9, legend=False,
                                                                        sharex='col', title=ordered_countries_new[x])

fig.legend(['Reported', 'Unreported'], 
           loc='upper center', ncol=5, bbox_to_anchor = (0.5, -0.01), fontsize=15)
plt.show()
# -

# Here we can also look at reporting by country for multiple trials (although this will double count trials as the same trial can appear across multiple CTAs.

# +
total_multi = solo_merge[solo_merge.nca_count > 1][['nca', 'entered_year', 'results_dummy']].groupby(['nca', 'entered_year']).count().rename({'results_dummy': 'denominator'}, axis=1)

reported_multi = solo_merge[solo_merge.nca_count > 1][['nca', 'entered_year', 'results_dummy']].groupby(['nca', 'entered_year']).sum().rename({'results_dummy': 'numerator'}, axis=1)


# +
merged_multi = total_multi.join(reported_multi)
merged_multi['unreported'] = merged_multi.denominator - merged_multi.numerator

stacked_multi = merged_multi.drop('denominator', axis=1).stack().unstack(1)
# -

fig, ax = plt.subplots(figsize = (10,5))
plt.bar(range(2004,2021), stacked_multi.loc[pd.IndexSlice[:, 'numerator'], :].sum())
plt.bar(range(2004,2021), stacked_multi.loc[pd.IndexSlice[:, 'unreported'], :].sum(), bottom=stacked_multi.loc[pd.IndexSlice[:, 'numerator'], :].sum())
ax.set_xticks(range(2004,2021))
ax.legend(['Results Available', 'Results Unavailable'], 
           loc='upper left', fontsize=8Mult)
plt.title('Reporting by Multiple CTA Trials', pad=10)
ax.set_axisbelow(True)
ax.grid(axis='y', zorder=0)
plt.show()

# +
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4)
fig.suptitle("Proportion of Multi-CTA Trials Reported by Year", y=1.05, fontsize=20)
fig.tight_layout()
for x, y in enumerate(fig.axes):
    stacked_multi.loc[[ordered_countries_new[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.9, legend=False,
                                                                        sharex='col', title=ordered_countries_new[x])

fig.legend(['Reported', 'Unreported'], 
           loc='upper center', ncol=5, bbox_to_anchor = (0.5, -0.01), fontsize=15)
plt.show()
# +



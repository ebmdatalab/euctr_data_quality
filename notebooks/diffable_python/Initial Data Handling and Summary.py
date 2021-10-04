# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + trusted=true
import pandas as pd

from lib.functions_data import nca_name_mapping, earliest_record_check

# + trusted=true
import sys
from pathlib import Path
import os
cwd = os.getcwd()
parent = str(Path(cwd).parents[0])
sys.path.append(parent)

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# -

# To avoid pulling the full dataset down each time we re-run the notebook, a CSV of the cut-down dataset is saved for easier reloading.

# + trusted=true
#Checking for the cut of the full dataset and creating it if it doesn't exist:
try:
    dec = pd.read_csv(parent + '/data/dec_euctr_extract.csv').drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    cols = ['eudract_number_with_country', 'date_of_competent_authority_decision', 
            'clinical_trial_type', 'national_competent_authority', 'eudract_number',
            'date_on_which_this_record_was_first_entered_in_the_eudract_data',
            'trial_status', 'date_of_the_global_end_of_the_trial', 'trial_results']

    #You can use this URL if you want to download the full raw data
    data_link = 'https://www.dropbox.com/s/4qt0msiipyn7crm/euctr_euctr_dump-2020-12-03-095517.csv.zip?dl=1'

    dec = pd.read_csv(data_link, compression='zip', low_memory=False, usecols=cols)
    dec.to_csv(parent + '/data/dec_euctr_extract.csv')

# + trusted=true
#Quick look at the spread of trial statuses on the EUCTR without any exclusions
dec.trial_status.value_counts(dropna=False)
# -

# The "date_of_competent_authority_decision" field has 2 nonsensical year values in which the correct value can reasonably be derived from context. We fix those below:
#
# https://www.clinicaltrialsregister.eu/ctr-search/trial/2009-016759-22/DK
#
# https://www.clinicaltrialsregister.eu/ctr-search/trial/2006-006947-30/FR

# + trusted=true
ind = dec[dec.date_of_competent_authority_decision.notnull() & 
          dec.date_of_competent_authority_decision.str.contains('210')].index
ind = ind.to_list()[0]

ind_2 = dec[dec.date_of_competent_authority_decision.notnull() & 
            dec.date_of_competent_authority_decision.str.contains('2077')].index
ind_2 = ind_2.to_list()[0]

dec.at[ind, 'date_of_competent_authority_decision'] = '2010-06-18'
dec.at[ind_2, 'date_of_competent_authority_decision'] = '2007-04-05'

# + trusted=true
#get rid of all protocols from non EU/EEA countries
dec_filt = dec[dec.clinical_trial_type != 'Outside EU/EEA'].reset_index(drop=True)

#lets see how many that is:
print(len(dec) - len(dec_filt))

# + trusted=true
dec_ctas = dec[['eudract_number', 'eudract_number_with_country']].groupby('eudract_number').count()['eudract_number_with_country']

print(f'There are {len(dec_ctas)} registered trials and {dec_ctas.sum()} CTAs including non-EU/EEA CTAs')

# + trusted=true
decf_ctas = dec_filt[['eudract_number', 'eudract_number_with_country']].groupby('eudract_number').count()['eudract_number_with_country']

print(f'There are {len(decf_ctas)} registered trials and {decf_ctas.sum()} CTAs excluding non-EU/EEA CTAs')

# + trusted=true
#Making dates into dates and adding a column of just the "Year" for relevant dates

dec_filt['date_on_which_this_record_was_first_entered_in_the_eudract_data'] = pd.to_datetime(dec_filt['date_on_which_this_record_was_first_entered_in_the_eudract_data'])
dec_filt['entered_year'] = dec_filt['date_on_which_this_record_was_first_entered_in_the_eudract_data'].dt.year

dec_filt['date_of_competent_authority_decision'] = pd.to_datetime(dec_filt['date_of_competent_authority_decision'])
dec_filt['approved_year'] = dec_filt['date_of_competent_authority_decision'].dt.year

# + trusted=true
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

# + trusted=true
#Table 1
analysis_df[['nca', 'eudract_number_country']].groupby('nca').count()

# + trusted=true
#You can reproduce the data on the earliest registered protocol (For table 1) for each country by running 
#this function with the appropriate country abbreviation. For example, to get the date for Italy:

print(earliest_record_check(analysis_df, 'Italy - AIFA'))

#Uncomment this to get the date for all countries at once
#for abrev in country_abrevs.keys():
#    print(f'Country: {abrev}\nEarliest record date: {earliest_record_check(dec_filt, abrev)}')

# + trusted=true
analysis_df.head()

# + trusted=true
#To save the main analysis df for use by the other notebooks. Only need to run this once the first time.
#analysis_df.to_csv(parent + '/data/analysis_df.csv')

# + trusted=true
#lastly this is helpful to have the country names, both the original ones from the EUCTR and the 
#current ones, in ordered lists.
#We're going to make these available as variables to import in the lib file manually
#But you can see what they look like here:
ordered_countries_original = list(dec_filt.national_competent_authority.value_counts().index)
ordered_countries_new = list(analysis_df.nca.value_counts().index)
# -





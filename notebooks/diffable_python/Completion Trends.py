# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   orig_nbformat: 4
# ---

# + trusted=true
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lib.functions_data import ordered_countries_original, ordered_countries_new, is_ongoing, zero_out_dict

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

# + trusted=true
usecols=['eudract_number', 'nca', 'entered_year', 'trial_status']
status_df = pd.read_csv(parent + '/data/analysis_df.csv', usecols = usecols)
status_df['trial_status'] = status_df.trial_status.fillna('Missing')

status_group = status_df.groupby(['nca', 'entered_year', 'trial_status'], as_index=False).count()

# + trusted=true
ordered_countries = list(status_group[['nca', 'eudract_number']].groupby('nca').sum().sort_values(by='eudract_number', ascending=False).index)

#Removing these for low number of trials
ordered_countries.remove('Malta - ADM')
ordered_countries.remove('Luxembourg - Ministry of Health')
ordered_countries.remove('Cyprus - MoH-Ph.S')

# + trusted=true
#Collapsing the various trial status categories on the EUCTR into simpler categories
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

# + trusted=true
#Shaping up the final data so it's easy to use

regrouped = pd.DataFrame.from_dict(country_status, orient='index').stack().to_frame()[0].apply(pd.Series).reindex(
    ['completed', 'ongoing', 'other', 'missing'], level=1)
# -

# # Overall Trend in Completion Status - Supplemental Figure 6

# + trusted=true
#A glance at the overall trend

grouped_total = regrouped.droplevel(level=0).groupby(regrouped.droplevel(level=0).index).sum()

title='Trend in Trial Status of Registered Protocols'


fig, ax = plt.subplots(figsize = (10,5), dpi=400)
grouped_total.reindex(['completed', 'ongoing', 'missing', 'other']).T.plot.bar(stacked=True, width=.9, ax=ax, 
                                                                               rot=45, title = title, 
                                                                               legend=False)

ax.set_axisbelow(True)
ax.grid(axis='y', zorder=0)
ax.set_xlabel('Record Entry Year', labelpad=10)
ax.set_ylabel('Record Count')

plt.legend(['Completed', 'Ongoing', 'Other', 'Missing'], 
           loc='upper center', ncol=5, bbox_to_anchor = (0.5, -0.2), fontsize=12)
plt.show()
#fig.savefig(parent + '/data/Figures/fig_s6.jpg', bbox_inches='tight', dpi=400)
#Saved
# -

# # Key Statistics

# + trusted=true
#What is the distribution of completed trials by year?

g_total = grouped_total.T
g_total['prct'] = round((g_total['completed'] / (g_total['ongoing'] + g_total['completed'] + g_total['missing'] + g_total['other'])) * 100,2)
g_total

# + trusted=true
#What percent of all protocols are in a completed status

print(f'There are {grouped_total.sum(axis=1)["completed"]} completed protocols')
print(f'There are {grouped_total.sum(axis=1).sum()} total protocols')
print(f'{round((grouped_total.sum(axis=1)["completed"]/grouped_total.sum(axis=1).sum())*100,2)}% are completed')

# + trusted=true
#What is the distribution of completed trials by country?

overall_prct_dict = {}

for x in ordered_countries:
    g = regrouped.loc[[x]].droplevel(level=0).T
    num = g.completed.sum()
    denom = num + g.ongoing.sum() + g.missing.sum() + g.other.sum()
    overall_prct_dict[x] = num / denom

rankings_completed = pd.Series(overall_prct_dict).sort_values(ascending=False)
rankings_completed

# + trusted=true
print(f'{round((len(rankings_completed[rankings_completed > .652]) / len(rankings_completed)) * 100,2)}% of countried exceed the overall mark')
# -

# # Status trends for every country - Supplemental Figure 7

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=400)
fig.suptitle("Trial Status Trends by Country Regulator", y=1.02, fontsize=23)
fig.tight_layout()
for x, y in enumerate(fig.axes):
    regrouped.loc[[rankings_completed.index[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.85, 
                                                                               legend=False, sharex='col', rot=45)
    
    y.set_title(rankings_completed.index[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(axis='y', zorder=0)
    y.set_xlabel('')
    
    hatches = ['','//', 'oo', '\\\\']
    counter = 0
    h_counter = 0
    patch_count = len(y.patches)
    for p in y.patches:
        p.set_hatch(hatches[h_counter])
        counter += 1
        if counter == ((patch_count/4) * (h_counter+1)):
            h_counter += 1
            

fig.text(-0.015, 0.5, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -0.025, 'Record Entry Year', ha='center', va='center', fontsize=20)

plt.legend(['Completed', 'Ongoing', 'Other', 'Missing'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.55), fontsize=15)

plt.show()
fig.savefig(parent + '/data/Figures/fig_s7.jpg', bbox_inches='tight', dpi=400)
#Saved
# -

# # Figure 4a
#
# Now limiting it just to the examples we are pulling out into the paper.

# + trusted=true
#Pull out the highlighted countries

fig, axes = plt.subplots(figsize = (20, 3), nrows=1, ncols=4, dpi=400)
fig.suptitle("(A)", y=1.04, x=0, fontsize=25)
fig.tight_layout()
included_countries = ['Lithuania - VVKT', 'Belgium - FAMHP', 'Netherlands - CCMO', 'Spain - AEMPS']

for x, y in enumerate(fig.axes):
    regrouped.loc[[included_countries[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.85, 
                                                                               legend=False, sharex='col', rot=45)
    
    y.set_title(included_countries[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(axis='y', zorder=0)
    y.set_xlabel('')
    
    hatches = ['','//', 'oo', '\\\\']
    counter = 0
    h_counter = 0
    patch_count = len(y.patches)
    for p in y.patches:
        p.set_hatch(hatches[h_counter])
        counter += 1
        if counter == ((patch_count/4) * (h_counter+1)):
            h_counter += 1
            

fig.text(-0.005, 0.4, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -0.13, 'Record Entry Year', ha='center', va='center', fontsize=20)

plt.legend(['Completed', 'Ongoing', 'Other', 'Missing'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.45), fontsize=15)


plt.show()
#fig.savefig(parent + '/data/Figures/fig_4a.jpg', bbox_inches='tight', dpi=400)
#fig.savefig(parent + '/data/Figures/fig_4a.eps', bbox_inches='tight', dpi=400)
#Saved
# -

# # Stats on conflicting and problematic trial status data for the paper

# + trusted=true
status_df['trial_status'].value_counts()

# + trusted=true
status_summary = status_df[['eudract_number', 'trial_status']].groupby('eudract_number')['trial_status'].count().to_frame(name='count').join(status_df[['eudract_number', 'trial_status']].groupby('eudract_number')['trial_status'].apply(list).to_frame(name='status'))
status_summary['set'] = status_summary['status'].apply(set)
multi_status = status_summary[status_summary['count'] > 1].reset_index().set_index('eudract_number')
indiv_status = multi_status['set'].to_list()

# + trusted=true
#This counts the number of trials that have an ongoing and a completed status

c = 0
indicator_var = []

#Looking for our "Ongoing" or "Completed" statuses
for i in indiv_status:
    if ('Ongoing' in i or 'Restarted' in i) and ('Completed' in i or 'Premautrely Ended' in i):
        c+=1
        indicator_var.append(1)
    else:
        indicator_var.append(0)
        
print(c)

# + trusted=true
print(f'{round((c/len(indiv_status)) * 100,2)}% of multi-protocol trials are in a conflicted status')

# + trusted=true
group_year = status_df[['eudract_number', 'entered_year']].groupby('eudract_number').max()

multi_status['conflict'] = indicator_var

year_joined = multi_status.join(group_year, how='left')

year_joined.head()

# + trusted=true
conflict_summary = year_joined[['conflict', 'entered_year']].reset_index(drop=True).groupby('entered_year').agg(['sum', 'count'])

conflict_summary['prct'] = round((conflict_summary['conflict']['sum'] / conflict_summary['conflict']['count'])*100,2)

conflict_summary.head()
# -

# # Supplemental Figure 8

# + trusted=true
fig, ax = plt.subplots(figsize = (10,5), dpi=400)

plt.plot(conflict_summary.index, conflict_summary['prct'], marker='.', markersize=10)
plt.grid()
plt.xlabel('Record Entry Year', labelpad=10)
plt.ylabel('Percent Conflicted')
plt.title('Trend in Conflicting Completion Information in Multi-Protocol Trials', pad=10)

ax.set_yticks(range(0, 101,10))
ax.set_xticks(range(2004,2021))

plt.show()
#fig.savefig(parent + '/data/Figures/fig_s8.jpg', bbox_inches='tight', dpi=400)
#Saved
# -

# What about trials with only a single protocol?

# + trusted=true
single_status = status_summary[status_summary['count'] == 1].reset_index().set_index('eudract_number').join(status_df.set_index('eudract_number')[['entered_year']], how='left')

single_status['ongoing'] = single_status['status'].apply(is_ongoing)

# + trusted=true
single_s_grouped = single_status[['count', 'ongoing', 'entered_year']].groupby('entered_year').sum()

# + trusted=true
#How many trials from prior to 2015 are still in an ongoing status?
print(single_s_grouped[single_s_grouped.index < 2015].sum())
print(round((6086/16552)*100,2))
# -

# # Completion date availability trends

# + trusted=true
usecols=['eudract_number', 'nca', 'entered_year', 'trial_status', 'completion_date', 'trial_results']
date_df = pd.read_csv(parent + '/data/analysis_df.csv', usecols = usecols)
date_df['trial_status'] = date_df.trial_status.fillna('Missing')
date_df['has_completion_date'] = np.where(date_df.completion_date.isna(), 0, 1)
only_completed = date_df[date_df.trial_status.isin(['Completed', 'Prematurely Ended'])].reset_index(drop=True)

# + trusted=true
total_completed = only_completed[['nca', 
                                  'entered_year', 
                                  'has_completion_date']].groupby(['nca', 
                                                                   'entered_year']).count().rename({'has_completion_date': 'denominator'}, axis=1)

total_completed_date = only_completed[['nca', 'entered_year', 'has_completion_date']].groupby(['nca', 'entered_year']).sum().rename({'has_completion_date': 'numerator'}, axis=1)

# + trusted=true
merged_dates = total_completed.join(total_completed_date)
merged_dates['missing_dates'] = merged_dates.denominator - merged_dates.numerator

stacked_dates = merged_dates.drop('denominator', axis=1).stack().unstack(1)
# -

# # Supplemental Figure 9

# + trusted=true
overall_dates = stacked_dates.droplevel(level=0).groupby(stacked_dates.droplevel(level=0).index).sum()

title='Availability of Completion Dates for Completed Protocols'

fig, ax = plt.subplots(figsize = (10,5), dpi=400)
overall_dates.reindex(['numerator', 'missing_dates']).T.plot.bar(stacked=True, width=.9, ax=ax, legend=False, 
                                                                 rot=45, title=title)

ax.set_axisbelow(True)
ax.grid(axis='y', zorder=0)
ax.set_xlabel('Record Entry Year', labelpad=10)
ax.set_ylabel('Record Count')

plt.legend(['Has Date', 'Missing Date'], 
           loc='upper right', fontsize=12)
plt.show()
#fig.savefig(parent + '/data/Figures/fig_s9.jpg', bbox_inches='tight', dpi=400)
#Saved

# + trusted=true
dates_trans = overall_dates.T
dates_trans['prct'] = dates_trans['numerator'] / (dates_trans['numerator'] + dates_trans['missing_dates'])

# + trusted=true
print(dates_trans.missing_dates.sum() + dates_trans.numerator.sum())
print(dates_trans.numerator.sum() / (dates_trans.missing_dates.sum() + dates_trans.numerator.sum()))

# + trusted=true
overall_comp_dict = {}

for x in ordered_countries:
    d = stacked_dates.loc[x].T
    num = d.numerator.sum()
    denom = num + d.missing_dates.sum()
    overall_comp_dict[x] = num / denom

rankings_compdate = pd.Series(overall_comp_dict).sort_values(ascending=False)
rankings_compdate
# -

# # Trends in completion data availability by country - Supplemental Figure 10

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=400)
fig.suptitle("Trends in Completion Date Availability by National Regulator", y=1.02, fontsize=23)
fig.tight_layout()

for x, y in enumerate(fig.axes):
    stacked_dates.loc[[rankings_compdate.index[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.85, 
                                                                            legend=False, sharex='col', rot=45)

    y.set_title(rankings_compdate.index[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(axis='y', zorder=0)
    y.set_xlabel('')
    
    hatches = ['','//']
    counter = 0
    h_counter = 0
    patch_count = len(y.patches)
    for p in y.patches:
        p.set_hatch(hatches[h_counter])
        counter += 1
        if counter == ((patch_count/2) * (h_counter+1)):
            h_counter += 1
    
fig.text(-0.015, 0.5, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -0.02, 'Record Entry Year', ha='center', va='center', fontsize=20)
    
plt.legend(['Has Date', 'Missing Date'], 
           loc='lower center', ncol=5, bbox_to_anchor = (-1.25, -.9), fontsize=15)

plt.show()
fig.savefig(parent + '/data/Figures/fig_s10.jpg', bbox_inches='tight', dpi=400)
#Saved
# -

# # Figure 4b

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 3), nrows=1, ncols=4, dpi=400)
fig.suptitle("(B)", y=1.04, x=0, fontsize=25)
fig.tight_layout()
included_countries = ['Lithuania - VVKT', 'Belgium - FAMHP', 'Netherlands - CCMO', 'Spain - AEMPS']

for x, y in enumerate(fig.axes):
    stacked_dates.loc[[included_countries[x]]].droplevel(level=0).T.plot.bar(stacked=True, ax=y, width=.85, 
                                                                            legend=False, sharex='col', rot=45)

    y.set_title(included_countries[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(axis='y', zorder=0)
    y.set_xlabel('')
    
    hatches = ['','//']
    counter = 0
    h_counter = 0
    patch_count = len(y.patches)
    for p in y.patches:
        p.set_hatch(hatches[h_counter])
        counter += 1
        if counter == ((patch_count/2) * (h_counter+1)):
            h_counter += 1
    
fig.text(-.011, 0.4, 'Completed\n Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -.087, 'Record Entry Year', ha='center', va='center', fontsize=20)
    
plt.legend(['Has Date', 'Missing Date'], 
           loc='lower center', ncol=5, bbox_to_anchor = (-1.2, -.7), fontsize=15)

plt.show()
#fig.savefig(parent + '/data/Figures/fig_4b.jpg', bbox_inches='tight', dpi=400)
#fig.savefig(parent + '/data/Figures/fig_4b.eps', bbox_inches='tight', dpi=400)
#Saved
# -

# # Combining Status and Completion Date (Figure 3)

# + trusted=true
usecols2 = ['eudract_number', 'nca', 'entered_year', 'trial_status', 'completion_date', 'trial_results']

# + trusted=true
usecols2 = ['eudract_number', 'nca', 'entered_year', 'trial_status', 'completion_date', 'trial_results']

date_df2 = pd.read_csv(parent + '/data/analysis_df.csv', usecols=usecols2)
date_df2['trial_status'] = date_df2.trial_status.fillna('Missing')
date_df2['has_completion_date'] = np.where(date_df2.completion_date.isna(), 0, 1)

# + trusted=true
total_trials = date_df2[['nca', 
                         'entered_year', 
                         'has_completion_date']].groupby(['nca', 
                                                          'entered_year']).count().rename({'has_completion_date': 'denominator'}, axis=1)

total_trials_date = only_completed[['nca', 'entered_year', 'has_completion_date']].groupby(['nca', 'entered_year']).sum().rename({'has_completion_date': 'numerator'}, axis=1)

merged_dates2 = total_trials.join(total_trials_date)
merged_dates2['missing_dates'] = merged_dates2.denominator - merged_dates2.numerator

stacked_dates2 = merged_dates2.drop('denominator', axis=1).stack().unstack(1)
stacked_dates2.head()

# + trusted=true
total_comp_dict = {}

for x in ordered_countries:
    d = stacked_dates2.loc[x].T
    num = d.numerator.sum()
    denom = num + d.missing_dates.sum()
    total_comp_dict[x] = num / denom

rankings_compdate2 = pd.Series(total_comp_dict).sort_values(ascending=False)

# + trusted=true
props_for_graph = rankings_completed.to_frame(name='prct_completed').join(rankings_compdate2.to_frame(name='prct_date'))
props_for_graph['diff1'] = props_for_graph.prct_completed - props_for_graph.prct_date
props_for_graph['diff2'] = 1 - props_for_graph.prct_completed

# + trusted=true
fig = plt.figure(figsize = (10,8), dpi=400)
ax = fig.add_subplot(111)

ax.barh(props_for_graph.index, props_for_graph.prct_date, align='center', height=.5, color='C0', hatch='////', label='Completed & Completion Date')
ax.barh(props_for_graph.index, props_for_graph.diff1, align='center', height=.5, color='C0', left=props_for_graph.prct_date, label='Completed & No Date')
ax.barh(props_for_graph.index, props_for_graph.diff2, align='center', height=.5, color='C1', left=props_for_graph.prct_completed, label = 'Ongoing and Other')
ax.set_xlabel('Proportion of Registered Protocols')
ax.tick_params(axis='x', labelsize=9)

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=3, loc='lower center', bbox_to_anchor = (.5, -.12))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

#plt.title('')
plt.tight_layout()
plt.show()
#fig.savefig(parent + '/data/Figures/fig_3.jpg', bbox_inches='tight', dpi=400)
#fig.savefig(parent + '/data/Figures/fig_3.eps', bbox_inches='tight', dpi=400)
#Saved
# -
# # Stat on conflicting dates

# + trusted=true
grouped_dates_count = date_df[['eudract_number', 'completion_date']].groupby('eudract_number').agg({'eudract_number': ['count'], 'completion_date': ['nunique', 'count']})

# + trusted=true
has_dates = grouped_dates_count[grouped_dates_count.completion_date['count'] > 1]

# + trusted=true
#Filters
all_have_dates = (has_dates.eudract_number['count'] == has_dates.completion_date['count'])
consistant_dates = has_dates.completion_date['nunique'] == 1

# + trusted=true
print(f'{round((len(has_dates[all_have_dates & consistant_dates])/len(has_dates)) * 100,2)}% of all trials with multiple protocols and at least one completion date are consistent across all protocols')
# +



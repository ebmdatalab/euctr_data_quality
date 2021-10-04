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
import matplotlib.pyplot as plt
import numpy as np

from lib.functions_data import ordered_countries_original, ordered_countries_new, zero_out_dict

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
usecols = ['eudract_number', 'nca', 'date_entered', 'entered_year', 'approved_date', 'approved_year']

reg_df = pd.read_csv(parent + '/data/analysis_df.csv', usecols = usecols)
reg_df['approved_date'] = pd.to_datetime(reg_df['approved_date'])
reg_df['date_entered'] = pd.to_datetime(reg_df['date_entered'])
reg_df.head()

# + trusted=true
#Data for Overall Trend in Registrations

grouped_overall = reg_df[['eudract_number']].groupby([reg_df.entered_year]).count()
earliest_entered = reg_df[['eudract_number', 'date_entered']].groupby('eudract_number', as_index=False).min()
earliest_entered['year'] = earliest_entered.date_entered.dt.year
unique_trials = earliest_entered[['eudract_number', 'year']].groupby('year').count()

# + trusted=true
grouped_overall.head()

# + trusted=true
#On the first run, I need to save the grouped_overall data to make it available to other notebooks
#grouped_overall.to_csv(parent + '/data/grouped_overall.csv')
# -

# # Supplemental Figue 1 - Overall registration trend

# + trusted=true
fig, ax = plt.subplots(figsize = (12,6), dpi=400)

grouped_overall[(grouped_overall.index > 2004) & (grouped_overall.index < 2020)].plot(ax=ax, legend=False, lw=2, 
                                                                                      marker='.', markersize=12)
unique_trials[(unique_trials.index > 2004) & (unique_trials.index < 2020)].plot(ax=ax, legend=False, grid=True, 
                                                                                lw=2, marker='^', markersize=10)

ax.legend(['Country Protocols', 'Unique Trials'], bbox_to_anchor = (1, 1))
ax.set_xticks(range(2005, 2020))
ax.set_yticks(range(0,7500, 500))
plt.xlabel('Protocol Entry Year', labelpad=10)
plt.ylabel('Records Entered')
plt.title('Trend in New Protocol and Overall Trial Registration on the EUCTR', pad=10)

#fig.savefig(parent + '/data/Figures/fig_s1.jpg', bbox_inches='tight', dpi=400)
#Saved
fig.show()
# -

# Now we can look at trends by country. You can group the data by any time period. I'm showing Quarter and Year below but you could do month, week, etc. by changing the "freq" parameter. First we prepare the data.

# + trusted=true
#Grouping by quarters - which we use for the cumulative graphs

grouped = reg_df[['eudract_number']].groupby([reg_df.nca, pd.PeriodIndex(reg_df.date_entered, freq='Q')]).count()

get_index = reg_df[['eudract_number']].groupby(pd.PeriodIndex(reg_df.date_entered, freq='Q')).count()
quarters = list(get_index.index)

# + trusted=true
#Grouping by years - which we use for the raw trend graphs

grouped_2 = reg_df[['eudract_number']].groupby([reg_df.nca, pd.PeriodIndex(reg_df.date_entered, freq='Y')]).count()

get_index = reg_df[['eudract_number']].groupby(pd.PeriodIndex(reg_df.date_entered, freq='Y')).count()
years = list(get_index.index)

# + trusted=true
#We are choosing to use "entered_year" as a variable to track trends over time, as this represents the date 
#the registration information was entered by the NCA. Just doing a quick validation against another
#date provided, the date of the NCA approval, to make sure they roughl align. This prepares that data.

grouped_year = reg_df[['eudract_number']].groupby([reg_df.nca, reg_df.entered_year]).count()
grouped_year_2 = reg_df[['eudract_number']].groupby([reg_df.nca, reg_df.approved_year]).count()
# -

# # Cumulative trend in new trials - Supplemental Figure 2
#
# Here we create a cumulative trend line for all NCAs and compare it to a trend line that assumes a constant rate of new trial per year.

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=400)
fig.suptitle("Cumulative Trial Protocol Registrations by National Regulatory Authority", y=1.02, fontsize=23)
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
    cumulative.plot(ax=y, lw=4, sharex='col',legend=False)
    
    #Plotting the reference line
    cumulative.loc[[cumulative.index[0], cumulative.index[-1]]].plot(ax=y, legend=False, lw=2, style='--')
    
    y.set_title(ordered_countries_new[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('')
    y.set_xlim(x_ticks[0], x_ticks[-1])
    
pd.set_option('mode.chained_assignment', 'warn')

fig.text(-0.015, 0.5, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -0.02, 'Record Entry Year', ha='center', va='center', fontsize=20)

plt.legend(['Cumulative Count of New Protocol Registrations', 'Stable Trend Line'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.55), fontsize=15)
plt.show()
#fig.savefig(parent + '/data/Figures/fig_s2.jpg', bbox_inches='tight', dpi=400)
#Saved
# -

# # Figure 2A
#
# For the paper, we do the same thing but limit the countries in the figure to only those of interest based on the results of the protocol availability section. Interested readers can check the supplement or this code for the trend across all countries.

# + trusted=true
#Reduced Figure
fig, axes = plt.subplots(figsize = (20, 3), nrows=1, ncols=5, dpi=400)
fig.suptitle("(A)", y=1.05, x=0, fontsize=25)
fig.tight_layout()

included_countries = ['UK - MHRA', 'France - ANSM', 'Norway - NoMA', 'Romania - ANMDM', 'Italy - AIFA']

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    country = grouped.loc[included_countries[x]]
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
    cumulative.plot(ax=y, lw=4, sharex='col',legend=False)
    
    #Plotting the reference line
    cumulative.loc[[cumulative.index[0], cumulative.index[-1]]].plot(ax=y, legend=False, lw=2, style='--')
    
    y.set_title(included_countries[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('')
    y.set_xlim(x_ticks[0], x_ticks[-1])
    
pd.set_option('mode.chained_assignment', 'warn')

fig.text(-0.015, 0.45, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=18)
fig.text(.5, -0.04, 'Record Entry Year', ha='center', va='center', fontsize=18)

plt.legend(['Cumulative Protocol Registrations', 'Stable Trend Line'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.9, -.34), fontsize=15)
plt.show()
#fig.savefig(parent + '/data/Figures/fig_2A.jpg', bbox_inches='tight', dpi=400)
#fig.savefig(parent + '/data/Figures/fig_2A.eps', bbox_inches='tight', dpi=400)
#Saved
# -

# # Annual trend in new trials - Supplemental Figure 3
#
# Here we will look at the same trends but not cumulatively, first for all countries then for just the ones we included in the reduced figure above which I will combine into a single figure for the paper. I'll also provide commented out code that people can switch to if they are interested in the data on new trials by Quarter, but this is a bit too messy for the paper.

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=400)
fig.suptitle("National Trends in Annual Trial Protocol Registrations", y=1.02, fontsize=23)
fig.tight_layout()

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    #For quarters, replace relevant variables with commented code:
    country = grouped_2.loc[ordered_countries_new[x]]
    #country = grouped.loc[ordered_countries_new[x]]
    first_reporting_year = country[country.eudract_number > 0].index.min()
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], years)
    #adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], quarters) 
    data = pd.DataFrame({'eudract_number': adjusted_data})
    x_ticks = data.index
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_year, np.nan, data.eudract_number)
    
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
    
    consolidated.plot(ax=y, lw=2, sharex='col',legend=False)
    
    y.set_title(ordered_countries_new[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('')
    y.set_xlim(x_ticks[0], x_ticks[-1])
    y.set_ylim(ymin=0)

    
    #y.xaxis.set_major_locator(MaxNLocator(prune='both'))    

pd.set_option('mode.chained_assignment', 'warn')

fig.text(-0.015, 0.5, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -0.02, 'Record Entry Year', ha='center', va='center', fontsize=20)

plt.show()
#fig.savefig(parent + '/data/Figures/fig_s3.jpg', bbox_inches='tight', dpi=400)
#Saved
# -

# # Figure 2b
#
# Same countries as Figure 2A

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 3), nrows=1, ncols=5, dpi=400)
fig.suptitle("(B)", y=1.05, x=0, fontsize=25)
fig.tight_layout()

included_countries = ['UK - MHRA', 'France - ANSM', 'Norway - NoMA', 'Romania - ANMDM', 'Italy - AIFA']

pd.set_option('mode.chained_assignment', None)
for x, y in enumerate(fig.axes):
    #For quarters, replace relevant variables with commented code:
    country = grouped_2.loc[included_countries[x]]
    #country = grouped.loc[ordered_countries_new[x]]
    first_reporting_year = country[country.eudract_number > 0].index.min()
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], years)
    #adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], quarters)
    data = pd.DataFrame({'eudract_number': adjusted_data})
    x_ticks = data.index
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_year, np.nan, data.eudract_number)
    
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
    
    consolidated.plot(ax=y, lw=2, sharex='col',legend=False)
    
    #if ordered_countries_original[x] == 'Slovenia - JAZMP':
    #    y.set_yticks(range(0,16,3))
    
    y.set_title(included_countries[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('')
    y.set_xlim(x_ticks[0], x_ticks[-1])
    y.set_ylim(ymin=0)
    y.set_xticks([pd.Period('2005', 'A-DEC'), pd.Period('2007', 'A-DEC'), pd.Period('2009', 'A-DEC'), 
                  pd.Period('2011', 'A-DEC'), pd.Period('2013', 'A-DEC'), pd.Period('2015', 'A-DEC'),
                  pd.Period('2017', 'A-DEC'), pd.Period('2019', 'A-DEC')])
    y.set_xticklabels(['2005', '2007', '2009', '2011', '2013', '2015', '2017', '2019'])
    
pd.set_option('mode.chained_assignment', 'warn')

fig.text(-0.015, 0.45, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=18)
fig.text(.5, -0.04, 'Record Entry Year', ha='center', va='center', fontsize=18)

plt.show()
#fig.savefig(parent + '/data/Figures/fig_2B.jpg', bbox_inches='tight', dpi=400)
#fig.savefig(parent + '/data/Figures/fig_2B.eps', bbox_inches='tight', dpi=400)
#Saved
# -

# # Comparison of different date variables - Supplemental Figure X
#
# This compared the annual trend by country using the "date_first_entere" and "date_approved" (meaning approved by the NCA) variables to make sure they aren't wildly different and would impact interpretation. While this date is generally smoother, they are aligned closely enough that no major interpretative differences should arise from using one vs. the other.

# + trusted=true
fig, axes = plt.subplots(figsize = (20, 16), nrows=7, ncols=4, dpi=400)
fig.suptitle("Comparison of Relevant Regulatory Dates", y=1.02, fontsize=23)
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
    
    consolidated.plot(ax=y, lw=2, sharex='col',legend=False)
    consolidated_2.plot(ax=y, lw=2, sharex='col',legend=False)
    
    y.set_title(ordered_countries_new[x], pad=6, fontsize=16)
    y.set_axisbelow(True)
    y.grid(zorder=0)
    y.set_xlabel('')
    y.set_xlim(x_ticks[0], x_ticks[-1])

fig.text(-0.015, 0.5, 'Record Count', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(.5, -0.02, 'Record Entry Year', ha='center', va='center', fontsize=20)
    
pd.set_option('mode.chained_assignment', 'warn')
plt.legend(['First Entered Date', 'NCA Approval Date'], 
           loc='upper center', ncol=5, bbox_to_anchor = (-1.2, -.5), fontsize=15)
plt.show()
#fig.savefig(parent + '/data/Figures/fig_s4.jpg', bbox_inches='tight', dpi=400)
#saved
# -











#Data
import pandas as pd
import ast
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np

#Data

cols = ['eudract_number', 
        'eudract_number_with_country', 
        'date_of_competent_authority_decision', 
        'national_competent_authority', 
        'date_on_which_this_record_was_first_entered_in_the_eudract_data', 
        'trial_status', 
        'date_of_the_global_end_of_the_trial', 
        'trial_results', 
        'clinical_trial_type']

country_abrevs = {'AT': 'Austria',
                'BE': 'Belgium',
                'BG': 'Bulgaria',
                'HR': 'Croatia',
                'CY': 'Cyprus',
                'CZ': 'Czech Republic',
                'DK': 'Denmark',
                'EE': 'Estonia',
                'FI': 'Finland',
                'FR': 'France',
                'DE': 'Germany',
                'GR': 'Greece',
                'HU': 'Hungary',
                'IS': 'Iceland',
                'IE': 'Ireland',
                'IT': 'Italy',
                'LV': 'Latvia',
                'LI': 'Liechtenstein',
                'LT': 'Lithuania',
                'LU': 'Luxembourg',
                'MT': 'Malta',
                'NL': 'Netherlands',
                'NO': 'Norway',
                'PL': 'Poland',
                'PT': 'Portugal',
                'RO': 'Romania',
                'SK': 'Slovakia',
                'SI': 'Slovenia',
                'ES': 'Spain',
                'SE': 'Sweden',
                'GB': 'United Kingdom'}


first_record_date = {'Austria': pd.Timestamp(2004,9,23), 
                     'Belgium': pd.Timestamp(2004,7,7), 
                     'Bulgaria': pd.Timestamp(2007,2,2), 
                     'Croatia': pd.Timestamp(2014,1,24), 
                     'Cyprus': pd.Timestamp(2009,2,24), 
                     'Czech Republic': pd.Timestamp(2004,6,24), 
                     'Denmark': pd.Timestamp(2004,8,10), 
                     'Estonia': pd.Timestamp(2004,11,26), 
                     'Finland': pd.Timestamp(2004,5,26), 
                     'France': pd.Timestamp(2005,6,21), 
                     'Germany': pd.Timestamp(2004,9,10), 
                     'Greece': pd.Timestamp(2005,11,4), 
                     'Hungary': pd.Timestamp(2004,6,15), 
                     'Iceland': pd.Timestamp(2004,9,7), 
                     'Ireland': pd.Timestamp(2004,6,18), 
                     'Italy': pd.Timestamp(2004,7,16), 
                     'Latvia': pd.Timestamp(2004,8,3), 
                     'Lithuania': pd.Timestamp(2004,6,22), 
                     'Luxembourg': pd.Timestamp(2013,7,26), 
                     'Malta': pd.Timestamp(2005,10,10), 
                     'Netherlands': pd.Timestamp(2006,3,16), 
                     'Norway': pd.Timestamp(2004,5,25), 
                     'Poland': pd.Timestamp(2007,3,29), 
                     'Portugal': pd.Timestamp(2005,8,18), 
                     'Romania': pd.Timestamp(2009,7,14), 
                     'Slovakia': pd.Timestamp(2004,6,2), 
                     'Slovenia': pd.Timestamp(2005,6,13), 
                     'Spain': pd.Timestamp(2004,6,14), 
                     'Sweden': pd.Timestamp(2004,5,13), 
                     'United Kingdom': pd.Timestamp(2004,7,1)}

nca_name_mapping = {'UK - MHRA': 'UK - MHRA', 
                    'Spain - AEMPS': 'Spain - AEMPS', 
                    'Germany - BfArM': 'Germany - BfArM',
                    'Italy - Italian Medicines Agency': 'Italy - AIFA', 
                    'Netherlands - Competent Authority': 'Netherlands - CCMO',
                    'France - ANSM': 'France - ANSM', 
                    'Belgium - FPS Health-DGM': 'Belgium - FAMHP', 
                    'Hungary - National Institute of Pharmacy': 'Hungary - OGYEI', 
                    'Czechia - SUKL': 'Czechia - SUKL',
                    'CZECHIA - SKUL': 'Czechia - SUKL',
                    'Austria - BASG': 'Austria - BASG', 
                    'Denmark - DHMA': 'Denmark - DKMA',
                    'Poland - Office for Medicinal Products': 'Poland - URPL', 
                    'Sweden - MPA': 'Sweden - MPA',
                    'Germany - PEI': 'Germany - PEI', 
                    'Finland - Fimea': 'Finland - FIMEA', 
                    'Bulgarian Drug Agency': 'Bulgaria - BDA',
                    'Greece - EOF': 'Greece - EOF', 
                    'Slovakia - SIDC (Slovak)': 'Slovakia - SUKL',
                    'Portugal - INFARMED': 'Portugal - INFARMED', 
                    'Lithuania - SMCA': 'Lithuania - VVKT', 
                    'Ireland - HPRA': 'Ireland - HPRA',
                    'Latvia - SAM': 'Latvia - ZVA', 
                    'Estonia - SAM': 'Estonia - SAM', 
                    'Croatia - MIZ': 'Croatia - MIZ', 
                    'Norway - NOMA': 'Norway - NoMA',
                    'Slovenia - JAZMP': 'Slovenia - JAZMP', 
                    'Romania - National Agency for Medicines and Medical Devices': 'Romania - ANMDM',
                    'Iceland - IMCA': 'Iceland - IMA'}


ordered_countries_original = ['UK - MHRA',
                              'Spain - AEMPS',
                              'Germany - BfArM',
                              'Italy - Italian Medicines Agency',
                              'France - ANSM',
                              'Belgium - FPS Health-DGM',
                              'Netherlands - Competent Authority',
                              'Hungary - National Institute of Pharmacy',
                              'Czech Republic - SUKL',
                              'Austria - BASG',
                              'Denmark - DHMA',
                              'Sweden - MPA',
                              'Poland - Office for Medicinal Products',
                              'Germany - PEI',
                              'Finland - Fimea',
                              'Bulgarian Drug Agency',
                              'Slovakia - SIDC (Slovak)',
                              'Greece - EOF',
                              'Portugal - INFARMED',
                              'Lithuania - SMCA',
                              'Ireland - HPRA',
                              'Latvia - SAM',
                              'Estonia - SAM',
                              'Norway - NOMA',
                              'Croatia - MIZ',
                              'Slovenia - JAZMP',
                              'Romania - National Agency for Medicines and Medical Devices',
                              'Iceland - IMCA',
                              'Malta - ADM',
                              'Luxembourg - Ministry of Health',
                              'Cyprus - MoH-Ph.S']

ordered_countries_new = ['UK - MHRA',
                         'Spain - AEMPS',
                         'Germany - BfArM',
                         'Italy - AIFA',
                         'France - ANSM',
                         'Belgium - FAMHP',
                         'Netherlands - CCMO',
                         'Hungary - OGYEI',
                         'Czechia - SUKL',
                         'Austria - BASG',
                         'Denmark - DKMA',
                         'Sweden - MPA',
                         'Poland - URPL',
                         'Germany - PEI',
                         'Finland - FIMEA',
                         'Bulgaria - BDA',
                         'Greece - EOF',
                         'Slovakia - SUKL',
                         'Portugal - INFARMED',
                         'Lithuania - VVKT',
                         'Ireland - HPRA',
                         'Latvia - ZVA',
                         'Estonia - SAM',
                         'Norway - NoMA',
                         'Croatia - MIZ',
                         'Slovenia - JAZMP',
                         'Romania - ANMDM',
                         'Iceland - IMA',
                         'Malta - ADM',
                         'Luxembourg - Ministry of Health',
                         'Cyprus - MoH-Ph.S']


#Functions

def earliest_record_check(data, nca_name):
    df = data[['eudract_number', 'nca', 'date_entered']]
    country_data = df[df.nca == nca_name].reset_index(drop=True)
    country_data['date_entered'] = pd.to_datetime(country_data['date_entered'])
    return country_data.date_entered.min().date()


def zero_out_dict(a_dict, values):
    for y in values:
        if y in a_dict.keys():
            continue
        else:
            a_dict[y] = 0
    return dict(sorted(a_dict.items()))

def stationary_output(df, nca, ind, print_on=True, diff=False):
    country = df.loc[nca]
    first_reporting_quarter = country[country.eudract_number > 0].index.min()
    adjusted_data = zero_out_dict(country.to_dict()['eudract_number'], ind) 
    data = pd.DataFrame({'eudract_number': adjusted_data})
    
    #Get rid of leading zeros
    data['eudract_number'] = np.where(data.index < first_reporting_quarter, np.nan, data.eudract_number)
    
    consolidated = data[(data.index.year > 2004) & (data.index.year < 2020) & data.eudract_number.notnull()].reset_index(drop=True)

    if diff:
        consolidated['diff'] = consolidated['eudract_number'] - consolidated['eudract_number'].shift(1)
        if print_on:
            print(f'{nca}\n')
            print ('Results of Dickey-Fuller Test:\n')
            print(adf_test(consolidated['diff'].dropna()))
            print('\n')
            print('Results of KPSS Test:\n')
            print(kpss_test(consolidated['diff'].dropna()))
            print('\n')
            return None
    
    elif print_on and not diff:
        print(f'{nca}\n')
        print ('Results of Dickey-Fuller Test:\n')
        print(adf_test(consolidated))
        print('\n')
        print('Results of KPSS Test:\n')
        print(kpss_test(consolidated))
        print('\n')
        return None
    else:
        return adf_test(consolidated)['interpretation'], kpss_test(consolidated)['interpretation']

def compare_enrollment_registration(zipped_cats, offset=0):
    results_list = []

    for tid, p, r_c, s_d in zipped_cats:
        trial_dict = {}
        trial_dict['trial_id'] = tid
        prots = []
        
        for x in ast.literal_eval(p):
            if x == 'Outside EU/EEA':
                continue
            else:
                prots.append(country_abrevs[x])
        
        if not isinstance(r_c, float):
            rec = ast.literal_eval(r_c)
            rec_countries = list(rec.keys())
        else:
            continue
            
        to_check_rec = []
        date_excluded = []
        non_eu_excluded = []
        for r in rec_countries:
            if r in list(country_abrevs.values()) and s_d > first_record_date[r] + pd.DateOffset(months=offset):
                to_check_rec.append(r)
            elif r in list(country_abrevs.values()) and s_d <= first_record_date[r] + pd.DateOffset(months=offset):
                date_excluded.append(r)
            else:
                non_eu_excluded.append(r)
        
        trial_dict['protocols'] = p
        
        trial_dict['recruitment_countries'] = list(rec.keys())
        
        trial_dict['date_exclusions'] = date_excluded
        
        trial_dict['non_eu_excluded'] = non_eu_excluded
        
        unaccounted = set(to_check_rec) - set(prots)
        
        
        trial_dict['accounted'] = list(set(to_check_rec) - unaccounted)
        
        trial_dict['unaccounted'] = list(unaccounted)
        
        results_list.append(trial_dict)

    return results_list

def is_ongoing(x):
    if "Ongoing" in x:
        return 1
    else:
        return 0

#Please note, this is only tested to work on a Mac. May need adjustment to work on Windows 
#(i.e. removal or adjustment of of multiprocessing)

import os
os.getcwd()
os.chdir('/Users/nicholasdevito/Desktop/euctr_res_scrape')
os.getcwd()

from requests import get, ConnectionError
from bs4 import BeautifulSoup
import re
from time import time, sleep
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import requests_cache

requests_cache.install_cache()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

start_time = time()

def get_url(url):
    response = get(url, verify = False)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    return soup

url = 'https://www.clinicaltrialsregister.eu/ctr-search/search?query=&resultsstatus=trials-with-results&page1'
soup = get_url(url)

#gets max page number
number_of_pages = soup.find('div', {'class': 'margin-bottom: 6px;'})
max_page_link = str(number_of_pages.find_all('a')[-1])
max_page = re.findall(r'\d+', max_page_link)[0]
print(f"max_page: {max_page}")

pages = list(range(1,int(max_page)+1))

#setting up variables for page URLs
euctr_base_url = 'https://www.clinicaltrialsregister.eu'
euctr_results_search_page = '/ctr-search/search?query=&resultsstatus=trials-with-results&page='

def try_to_connect(tries, url):
    for i in range(tries):
        try:
            page_html = get_url(url)
            break
        except ConnectionError:
            if i < tries - 1:
                #print('Retrying {}...'.format(url))
                sleep(2 ** tries)
                continue 
            else:
                print(f'This URL failed: {url}')
                page_html = None
    return page_html

def get_trials(page):
    euctr_base_url = 'https://www.clinicaltrialsregister.eu'
    euctr_results_search_page = '/ctr-search/search?query=&resultsstatus=trials-with-results&page='
    
    #make the request
    page_html = try_to_connect(3, euctr_base_url + euctr_results_search_page + str(page))

    #get the tables
    trial_tables = page_html.find_all('table', {'class': 'result'})
    trial_info = []
    for trial_table in trial_tables:
        trial_id = trial_table.input.get('value')
        url = euctr_base_url + trial_table.find_all('a')[-1].get('href')
        trial_tuple = (trial_id, url)
        trial_info.append(trial_tuple)
    return trial_info


if __name__ == '__main__':
    with Pool() as p:
        trial_info = list(tqdm(p.imap(get_trials, pages), total=len(pages)))

tuples = list(trial_info)
tuples = [t for sublist in tuples for t in sublist]

print(f'There are {len(tuples)} trials')


def content_test(page_html):
    try:
        page_html.find('div', id = 'synopsisLegislationNote')
        page_html.find_all('table')[4]
        return True
    except(AttributeError, IndexError):
        return False

def get_results_info(tup):
    try:
        
        t_d = {}
        
        tries = 4
        for t in range(tries):
            page_html = try_to_connect(5, tup[1])
            if t < tries-1:
                if content_test(page_html):
                    break
                else:
                    #print('Reloading Content for {}'.format(tup[0]))
                    sleep(2 ** t)
                    continue
        try:
            leg_text = page_html.find('div', id = 'synopsisLegislationNote')
        except AttributeError:
            print(f'Error in trial {tup[0]}')
            t_d = {}
            t_d['trial_id'] = tup[0]
            t_d['error'] = 'Error in Scrape'
            return t_d
        try:
            trial_tables = page_html.find_all('table')[4]
        except IndexError:
            print(f'Error in trial {tup[0]}')
            t_d = {}
            t_d['trial_id'] = tup[0]
            t_d['error'] = 'Error in Scrape'
            return t_d
        
        
        #select all the results tables
        leg_text = page_html.find('div', id = 'synopsisLegislationNote')
        trial_tables = page_html.find_all('table')[4]
        
        t_d['trial_id'] = tup[0]
        
        try:
            global_end_date = trial_tables.find('td', text='    Global end of trial date').find_next('td').div.text.strip()
        except AttributeError:
            global_end_date = trial_tables.find('td', text='    Global completion date').find_next('td').div.text.strip()
        t_d['global_end_of_trial_date'] = global_end_date
        
        if trial_tables.find('td', text='    First version publication date'):
            t_d['first_version_date'] = trial_tables.find('td', text='    First version publication date').find_next('td').div.text.strip()
        else:
            t_d['first_version_date'] = None
        
        if trial_tables.find('td', text='    This version publication date'):
            t_d['this_version_date'] = trial_tables.find('td', text='    This version publication date').find_next('td').div.text.strip()
        else:
            t_d['this_version_date'] = None 
        
        country_prots = []
        pr_countries =  trial_tables.find('td', text='    Trial protocol').find_next('td').find_all('a')
        for c in pr_countries:
            country_prots.append(c.text.strip())
        t_d['trial_countries'] = country_prots
        
        if trial_tables.find('td', text='    Summary report(s)') and leg_text:
            t_d['results_type'] = 'Document'

        elif not trial_tables.find('td', text='    Summary report(s)') and leg_text:
            t_d['results_type'] = 'Broken'
            
        else:    
            trial_info = page_html.find_all('table')[5]
            
            rc_dict = {}
            rec_countries = trial_info.find_all('td', text='    Country: Number of subjects enrolled')
            for rc in rec_countries:
                country = rc.find_next('td').div.text.strip().split(':')
                rc_dict[country[0].strip()] = int(country[1].strip())
            t_d['recruitment_countries'] = rc_dict
            
            start_date = trial_info.find('td', text='    Actual start date of recruitment').find_next('td').div.text.strip()
            t_d['trial_start_date'] = start_date
            
            if trial_tables.find('td', text='    Summary report(s)') and not leg_text:
                t_d['results_type'] = "Mixed"
            else:
                t_d['results_type'] = "Tabular"
        return t_d
    except Exception as e:
        import sys
        raise type(e)(str(e) + 
                   ' happens at {}'.format(tup[0])).with_traceback(sys.exc_info()[2])

if __name__ == '__main__':
    with Pool() as p:
        results = list(tqdm(p.imap(get_results_info, tuples), total=len(tuples)))

results = list(results)

if len(tuples) == len(results):
    print("All Scraped")
else:
    print("Error in Scrape: Difference of {} trials between first and second scrape".format(len(tuples) - len(results)))

results_df = pd.DataFrame(results)

#dates to datetime
results_df['global_end_of_trial_date'] = pd.to_datetime(results_df['global_end_of_trial_date'])
results_df['first_version_date'] = pd.to_datetime(results_df['first_version_date'])
results_df['this_version_date'] = pd.to_datetime(results_df['this_version_date'])
results_df['trial_start_date'] = pd.to_datetime(results_df['trial_start_date'])

results_df.to_csv('euctr_data_quality_results_scrape_sept_2021.csv')

requests_cache.uninstall_cache()


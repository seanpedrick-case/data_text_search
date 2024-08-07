# ## Some functions to clean text
import re
import string

# Add calendar months onto stop words
import calendar

from typing import List

# Adding custom words to the stopwords
custom_words = []
my_stop_words = custom_words

cal_month = (list(calendar.month_name))
cal_month = [x.lower() for x in cal_month]

# Remove blanks
cal_month = [i for i in cal_month if i]
#print(cal_month)
custom_words.extend(cal_month)

# #### Some of my cleaning functions
replace_backslash = r'\\'
email_start_pattern_regex = r'.*importance:|.*subject:'
email_end_pattern_regex = r'kind regards.*|many thanks.*|sincerely.*'
html_pattern_regex = r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\xa0|&nbsp;'
email_pattern_regex = r'\S*@\S*\s?'
num_pattern_regex = r'[0-9]+'
postcode_pattern_regex = r'(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$)'
warning_pattern_regex = r'caution: this email originated from outside of the organization. do not click links or open attachments unless you recognize the sender and know the content is safe.'
nbsp_pattern_regex = r'&nbsp;'
multiple_spaces_regex = r'\s{2,}'

def initial_clean(texts:List[str]):
    """
    This function cleans a list of text strings by performing various replacements using polars.

    Args:
        texts (List[str]): A list of strings to clean.
        
    Returns:
        List[str]: A list of cleaned strings.
    """
    import polars as pl

    texts = pl.Series(texts)

    text = texts.str.replace_all(replace_backslash, '/')
    text = text.str.replace_all(html_pattern_regex, '')
    text = text.str.replace_all(email_start_pattern_regex, '')
    text = text.str.replace_all(email_end_pattern_regex, '')
    text = text.str.replace_all(email_pattern_regex, '')
    text = text.str.replace_all(multiple_spaces_regex, ' ')

    text = text.to_list()
    
    return text


def initial_clean_pandas(texts: List[str]):
    """
    This function cleans a list of text strings by performing various replacements using pandas.

    Args:
        texts (List[str]): A list of strings to clean.
        
    Returns:
        List[str]: A list of cleaned strings.
    """
    import pandas as pd

    # Create a pandas Series from the text list for easier manipulation
    text_series = pd.Series(texts)  
    
    # Replace patterns with pandas string methods (`.str.replace`)
    text_series = text_series.astype(str).str.replace(replace_backslash, '/', regex=True)
    text_series = text_series.astype(str).str.replace(html_pattern_regex, '', regex=True)
    text_series = text_series.astype(str).str.replace(email_start_pattern_regex, '', regex=True)
    text_series = text_series.astype(str).str.replace(email_end_pattern_regex, '', regex=True)
    text_series = text_series.astype(str).str.replace(email_pattern_regex, '', regex=True)
    text_series = text_series.astype(str).str.replace(multiple_spaces_regex, ' ', regex=True)
    
    # Convert cleaned Series back to a list
    return text_series.tolist()

def remove_hyphens(text_text):
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', text_text)


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

def convert_to_lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

def remove_short_tokens(tokens):
    return [token for token in tokens if len(token) > 3]


def remove_dups_text(data_samples_ready, data_samples_clean, data_samples):
   # Identify duplicates in the data: https://stackoverflow.com/questions/44191465/efficiently-identify-duplicates-in-large-list-500-000
    # Only identifies the second duplicate

    seen = set()
    dups = []

    for i, doi in enumerate(data_samples_ready):
        if doi not in seen:
            seen.add(doi)
        else:
            dups.append(i) 
    #data_samples_ready[dupes[0:]]
    
    # To see a specific duplicated value you know the position of
    #matching = [s for s in data_samples_ready if data_samples_ready[83] in s]
    #matching
    
    # Remove duplicates only (keep first instance)
    #data_samples_ready = list( dict.fromkeys(data_samples_ready) ) # This way would keep one version of the duplicates
    
    ### Remove all duplicates including original instance
    
    # Identify ALL duplicates including initial values
    # https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python

    from collections import defaultdict
    D = defaultdict(list)
    for i,item in enumerate(data_samples_ready):
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}
    
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    L = list(D.values())
    flat_list_dups = [item for sublist in L for item in sublist]

    # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
    for index in sorted(flat_list_dups, reverse=True):
        del data_samples_ready[index]
        del data_samples_clean[index]
        del data_samples[index]
    
    # Remove blanks
    data_samples_ready = [i for i in data_samples_ready if i]
    data_samples_clean = [i for i in data_samples_clean if i]
    data_samples = [i for i in data_samples if i]
    
    return data_samples_ready, data_samples_clean, flat_list_dups, data_samples


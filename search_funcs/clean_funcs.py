# ## Some functions to clean text

# ### Some other suggested cleaning approaches
#
# #### From here: https://shravan-kuchkula.github.io/topic-modeling/#interactive-plot-showing-results-of-k-means-clustering-lda-topic-modeling-and-sentiment-analysis
#
# - remove_hyphens
# - tokenize_text
# - remove_special_characters
# - convert to lower case
# - remove stopwords
# - lemmatize the token
# - remove short tokens
# - keep only words in wordnet
# - I ADDED ON - creating custom stopwords list

# +
# Create a custom stop words list
import nltk
import re
import string
import polars as pl
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

# Add calendar months onto stop words
import calendar
from tqdm import tqdm
import gradio as gr

stemmer = PorterStemmer()


nltk.download('stopwords')
nltk.download('wordnet')

#nltk.download('words')
#nltk.download('names')

#nltk.corpus.words.words('en')  

#from sklearn.feature_extraction import text
# Adding common names to stopwords

all_names = [x.lower() for x in list(nltk.corpus.names.words())]

# Adding custom words to the stopwords
custom_words = []
my_stop_words = custom_words


cal_month = (list(calendar.month_name))
cal_month = [x.lower() for x in cal_month]

# Remove blanks
cal_month = [i for i in cal_month if i]
#print(cal_month)
custom_words.extend(cal_month)
    
#my_stop_words = frozenset(text.ENGLISH_STOP_WORDS.union(custom_words).union(all_names))
#custom_stopwords = my_stop_words
# -

# #### Some of my cleaning functions
'''
# +
# Remove all html elements from the text. Inspired by this: https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string

def remove_email_start(text):
  cleanr = re.compile('.*importance:|.*subject:')
  cleantext = re.sub(cleanr, '', text)
  return cleantext

def remove_email_end(text):
  cleanr = re.compile('kind regards.*|many thanks.*|sincerely.*')
  cleantext = re.sub(cleanr, '', text)
  return cleantext
    
def cleanhtml(text):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\xa0')
  cleantext = re.sub(cleanr, '', text)
  return cleantext

## The above doesn't work when there is no > at the end of the string to match the initial <. Trying this: <[^>]+> but needs work: https://stackoverflow.com/questions/2013124/regex-matching-up-to-the-first-occurrence-of-a-character

# Remove all email addresses and numbers from the text

def cleanemail(text):
  cleanr = re.compile('\S*@\S*\s?|\xa0')
  cleantext = re.sub(cleanr, '', text)
  return cleantext

def cleannum(text):
  cleanr = re.compile(r'[0-9]+')
  cleantext = re.sub(cleanr, '', text)
  return cleantext

def cleanpostcode(text):
  cleanr = re.compile(r'(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$)')
  cleantext = re.sub(cleanr, '', text)
  return cleantext

def cleanwarning(text):
  cleanr = re.compile('caution: this email originated from outside of the organization. do not click links or open attachments unless you recognize the sender and know the content is safe.')
  cleantext = re.sub(cleanr, '', text)
  return cleantext


# -

def initial_clean(texts):
    clean_texts = []
    for text in texts:
        text = remove_email_start(text)
        text = remove_email_end(text)
        text = cleanpostcode(text)
        text = remove_hyphens(text)
        text = cleanhtml(text)
        text = cleanemail(text)
        #text = cleannum(text)        
        clean_texts.append(text)
    return clean_texts
'''

email_start_pattern_regex = r'.*importance:|.*subject:'
email_end_pattern_regex = r'kind regards.*|many thanks.*|sincerely.*'
html_pattern_regex = r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\xa0|&nbsp;'
email_pattern_regex = r'\S*@\S*\s?'
num_pattern_regex = r'[0-9]+'
postcode_pattern_regex = r'(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$)'
warning_pattern_regex = r'caution: this email originated from outside of the organization. do not click links or open attachments unless you recognize the sender and know the content is safe.'
nbsp_pattern_regex = r'&nbsp;'

# Pre-compiling the regular expressions for efficiency
email_start_pattern = re.compile(email_start_pattern_regex)
email_end_pattern = re.compile(email_end_pattern_regex)
html_pattern = re.compile(html_pattern_regex)
email_pattern = re.compile(email_end_pattern_regex)
num_pattern = re.compile(num_pattern_regex)
postcode_pattern = re.compile(postcode_pattern_regex)
warning_pattern = re.compile(warning_pattern_regex)
nbsp_pattern = re.compile(nbsp_pattern_regex)

def stem_sentence(sentence):

    words = sentence.split()
    stemmed_words = [stemmer.stem(word).lower().rstrip("'") for word in words]
    return stemmed_words

def stem_sentences(sentences, progress=gr.Progress()):
        """Stem each sentence in a list of sentences."""
        stemmed_sentences = [stem_sentence(sentence) for sentence in progress.tqdm(sentences)]
        return stemmed_sentences

def get_lemma_text(text):
    # Tokenize the input string into words
    tokens = word_tokenize(text)
    
    lemmas = []
    for word in tokens:
        if len(word) > 3:
            lemma = wn.morphy(word)
        else:
            lemma = None
        
        if lemma is None:
            lemmas.append(word)
        else:
            lemmas.append(lemma)
    return lemmas

def get_lemma_tokens(tokens):
    # Tokenize the input string into words
    
    lemmas = []
    for word in tokens:
        if len(word) > 3:
            lemma = wn.morphy(word)
        else:
            lemma = None
        
        if lemma is None:
            lemmas.append(word)
        else:
            lemmas.append(lemma)
    return lemmas

# def initial_clean(texts , progress=gr.Progress()):
#     clean_texts = []

#     i = 1
#     #progress(0, desc="Cleaning texts")
#     for text in progress.tqdm(texts, desc = "Cleaning data", unit = "rows"):
#         #print("Cleaning row: ", i)
#         text = re.sub(email_start_pattern, '', text)
#         text = re.sub(email_end_pattern, '', text)
#         text = re.sub(postcode_pattern, '', text)
#         text = remove_hyphens(text)  
#         text = re.sub(html_pattern, '', text)
#         text = re.sub(email_pattern, '', text)
#         text = re.sub(nbsp_pattern, '', text)
#         #text = re.sub(warning_pattern, '', text)
#         #text = stem_sentence(text)
#         text = get_lemma_text(text)
#         text = ' '.join(text)
#         # Uncomment the next line if you want to remove numbers as well
#         # text = re.sub(num_pattern, '', text)        
#         clean_texts.append(text)

#         i += 1
#     return clean_texts


def initial_clean(texts , progress=gr.Progress()):
    texts = pl.Series(texts)#[]

    #i = 1
    #progress(0, desc="Cleaning texts")
    #for text in progress.tqdm(texts, desc = "Cleaning data", unit = "rows"):
    #print("Cleaning row: ", i)
    text = texts.str.replace_all(email_start_pattern_regex, '')
    text = text.str.replace_all(email_end_pattern_regex, '')
    #text = re.sub(postcode_pattern, '', text)
    #text = remove_hyphens(text)  
    text = text.str.replace_all(html_pattern_regex, '')
    text = text.str.replace_all(email_pattern_regex, '')
    #text = re.sub(nbsp_pattern, '', text)
    #text = re.sub(warning_pattern, '', text)
    #text = stem_sentence(text)
    #text = get_lemma_text(text)
    #text = ' '.join(text)
    # Uncomment the next line if you want to remove numbers as well
    # text = re.sub(num_pattern, '', text)        
    #clean_texts.append(text)

    #i += 1

    text = text.to_list()
    
    return text


# Sample execution
#sample_texts = [
#    "Hello, this is a test email. kind regards, John",
#    "<div>Email content here</div> many thanks, Jane",
#   "caution: this email originated from outside of the organization. do not click links or open attachments unless you recognize the sender and know the content is safe.",
#    "john.doe123@example.com",
#    "Address: 1234 Elm St, AB12 3CD"
#]

#initial_clean(sample_texts)


# +

all_names = [x.lower() for x in list(nltk.corpus.names.words())]

def remove_hyphens(text_text):
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', text_text)

# tokenize text
def tokenize_text(text_text):
    TOKEN_PATTERN = r'\s+'
    regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=True)
    word_tokens = regex_wt.tokenize(text_text)
    return word_tokens

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

def convert_to_lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

def remove_stopwords(tokens, custom_stopwords):
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list += my_stop_words
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def remove_names(tokens):
    stopword_list = list(nltk.corpus.names.words())
    stopword_list = [x.lower() for x in stopword_list]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens



def remove_short_tokens(tokens):
    return [token for token in tokens if len(token) > 3]

def keep_only_words_in_wordnet(tokens):
    return [token for token in tokens if wn.synsets(token)]

def apply_lemmatize(tokens, wnl=WordNetLemmatizer()):

    def lem_word(word):
    
        if len(word) > 3: out_word = wnl.lemmatize(word)
        else: out_word = word

        return out_word

    return [lem_word(token) for token in tokens]


# +
### Do the cleaning

def cleanTexttexts(texts):
    clean_texts = []
    for text in texts:
        #text = remove_email_start(text)
        #text = remove_email_end(text)
        text = remove_hyphens(text)
        text = cleanhtml(text)
        text = cleanemail(text)
        text = cleanpostcode(text)
        text = cleannum(text)
        #text = cleanwarning(text)
        text_i = tokenize_text(text)
        text_i = remove_characters_after_tokenization(text_i)
        #text_i = remove_names(text_i)
        text_i = convert_to_lowercase(text_i)
        #text_i = remove_stopwords(text_i, my_stop_words)
        text_i = get_lemma(text_i)
        #text_i = remove_short_tokens(text_i)
        text_i = keep_only_words_in_wordnet(text_i)

        text_i = apply_lemmatize(text_i)
        clean_texts.append(text_i)
    return clean_texts


# -

def remove_dups_text(data_samples_ready, data_samples_clean, data_samples):
   # Identify duplicates in the data: https://stackoverflow.com/questions/44191465/efficiently-identify-duplicates-in-large-list-500-000
    # Only identifies the second duplicate

    seen = set()
    dupes = []

    for i, doi in enumerate(data_samples_ready):
        if doi not in seen:
            seen.add(doi)
        else:
            dupes.append(i) 
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


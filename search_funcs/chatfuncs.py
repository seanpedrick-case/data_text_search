import re
import os
from typing import TypeVar, List
import pandas as pd


# Model packages
import torch.cuda

# Alternative model sources
#from dataclasses import asdict, dataclass

# Langchain functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# For keyword extraction (not currently used)
#import nltk
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# For Name Entity Recognition model
#from span_marker import SpanMarkerModel # Not currently used


import gradio as gr

torch.cuda.empty_cache()

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

embeddings = None  # global variable setup
vectorstore = None # global variable setup
model_type = None # global variable setup

max_memory_length = 0 # How long should the memory of the conversation last?

full_text = "" # Define dummy source text (full text) just to enable highlight function to load

model = [] # Define empty list for model functions to run
tokenizer = [] # Define empty list for model functions to run

## Highlight text constants
hlt_chunk_size = 12
hlt_strat = [" ", ". ", "! ", "? ", ": ", "\n\n", "\n", ", "]
hlt_overlap = 4

## Initialise NER model ##
ner_model = []#SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd") # Not currently used


# Currently set gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
if torch.cuda.is_available():
    torch_device = "cuda"
    gpu_layers = 0
else: 
    torch_device =  "cpu"
    gpu_layers = 0

print("Running on device:", torch_device)
threads = 6 #torch.get_num_threads()
print("CPU threads:", threads)

# Vectorstore funcs

# Prompt functions

def write_out_metadata_as_string(metadata_in):
    metadata_string = [f"{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}" for d in metadata_in] # ['metadata']
    return metadata_string


def determine_file_type(file_path):
        """
        Determine the file type based on its extension.
    
        Parameters:
            file_path (str): Path to the file.
    
        Returns:
            str: File extension (e.g., '.pdf', '.docx', '.txt', '.html').
        """
        return os.path.splitext(file_path)[1].lower()


def create_doc_df(docs_keep_out):
    # Extract content and metadata from 'winning' passages.
            content=[]
            meta=[]
            meta_url=[]
            page_section=[]
            score=[]

            doc_df = pd.DataFrame()

            

            for item in docs_keep_out:
                content.append(item[0].page_content)
                meta.append(item[0].metadata)
                meta_url.append(item[0].metadata['source'])

                file_extension = determine_file_type(item[0].metadata['source'])
                if (file_extension != ".csv") & (file_extension != ".xlsx"):
                    page_section.append(item[0].metadata['page_section'])
                else: page_section.append("")
                score.append(item[1])       

            # Create df from 'winning' passages

            doc_df = pd.DataFrame(list(zip(content, meta, page_section, meta_url, score)),
               columns =['page_content', 'metadata', 'page_section', 'meta_url', 'score'])

            docs_content = doc_df['page_content'].astype(str)
            doc_df['full_url'] = "https://" + doc_df['meta_url'] 

            return doc_df


def get_expanded_passages(vectorstore, docs, width):

    """
    Extracts expanded passages based on given documents and a width for context.
    
    Parameters:
    - vectorstore: The primary data source.
    - docs: List of documents to be expanded.
    - width: Number of documents to expand around a given document for context.
    
    Returns:
    - expanded_docs: List of expanded Document objects.
    - doc_df: DataFrame representation of expanded_docs.
    """

    from collections import defaultdict
    
    def get_docs_from_vstore(vectorstore):
        vector = vectorstore.docstore._dict
        return list(vector.items())

    def extract_details(docs_list):
        docs_list_out = [tup[1] for tup in docs_list]
        content = [doc.page_content for doc in docs_list_out]
        meta = [doc.metadata for doc in docs_list_out]
        return ''.join(content), meta[0], meta[-1]
    
    def get_parent_content_and_meta(vstore_docs, width, target):
        #target_range = range(max(0, target - width), min(len(vstore_docs), target + width + 1))
        target_range = range(max(0, target), min(len(vstore_docs), target + width + 1)) # Now only selects extra passages AFTER the found passage
        parent_vstore_out = [vstore_docs[i] for i in target_range]
        
        content_str_out, meta_first_out, meta_last_out = [], [], []
        for _ in parent_vstore_out:
            content_str, meta_first, meta_last = extract_details(parent_vstore_out)
            content_str_out.append(content_str)
            meta_first_out.append(meta_first)
            meta_last_out.append(meta_last)
        return content_str_out, meta_first_out, meta_last_out

    def merge_dicts_except_source(d1, d2):
            merged = {}
            for key in d1:
                if key != "source":
                    merged[key] = str(d1[key]) + " to " + str(d2[key])
                else:
                    merged[key] = d1[key]  # or d2[key], based on preference
            return merged

    def merge_two_lists_of_dicts(list1, list2):
        return [merge_dicts_except_source(d1, d2) for d1, d2 in zip(list1, list2)]

    # Step 1: Filter vstore_docs
    vstore_docs = get_docs_from_vstore(vectorstore)
    doc_sources = {doc.metadata['source'] for doc, _ in docs}
    vstore_docs = [(k, v) for k, v in vstore_docs if v.metadata.get('source') in doc_sources]

    # Step 2: Group by source and proceed
    vstore_by_source = defaultdict(list)
    for k, v in vstore_docs:
        vstore_by_source[v.metadata['source']].append((k, v))
        
    expanded_docs = []
    for doc, score in docs:
        search_source = doc.metadata['source']
        

        #if file_type == ".csv" | file_type == ".xlsx":
        #     content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], 0, search_index)

        #else:
        search_section = doc.metadata['page_section']
        parent_vstore_meta_section = [doc.metadata['page_section'] for _, doc in vstore_by_source[search_source]]
        search_index = parent_vstore_meta_section.index(search_section) if search_section in parent_vstore_meta_section else -1

        content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], width, search_index)
        meta_full = merge_two_lists_of_dicts(meta_first, meta_last)

        expanded_doc = (Document(page_content=content_str[0], metadata=meta_full[0]), score)
        expanded_docs.append(expanded_doc)

    doc_df = pd.DataFrame()

    doc_df = create_doc_df(expanded_docs)  # Assuming you've defined the 'create_doc_df' function elsewhere

    return expanded_docs, doc_df

def highlight_found_text(search_text: str, full_text: str, hlt_chunk_size:int=hlt_chunk_size, hlt_strat:List=hlt_strat, hlt_overlap:int=hlt_overlap) -> str:
    """
    Highlights occurrences of search_text within full_text.
    
    Parameters:
    - search_text (str): The text to be searched for within full_text.
    - full_text (str): The text within which search_text occurrences will be highlighted.
    
    Returns:
    - str: A string with occurrences of search_text highlighted.
    
    Example:
    >>> highlight_found_text("world", "Hello, world! This is a test. Another world awaits.")
    'Hello, <mark style="color:black;">world</mark>! This is a test. Another <mark style="color:black;">world</mark> awaits.'
    """

    def extract_text_from_input(text, i=0):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[i][0].replace("  ", " ").strip()
        else:
            return ""

    def extract_search_text_from_input(text):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[-1][1].replace("  ", " ").strip()
        else:
            return ""

    full_text = extract_text_from_input(full_text)
    search_text = extract_search_text_from_input(search_text)



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=hlt_chunk_size,
        separators=hlt_strat,
        chunk_overlap=hlt_overlap,
    )
    sections = text_splitter.split_text(search_text)

    found_positions = {}
    for x in sections:
        text_start_pos = 0
        while text_start_pos != -1:
            text_start_pos = full_text.find(x, text_start_pos)
            if text_start_pos != -1:
                found_positions[text_start_pos] = text_start_pos + len(x)
                text_start_pos += 1

    # Combine overlapping or adjacent positions
    sorted_starts = sorted(found_positions.keys())
    combined_positions = []
    if sorted_starts:
        current_start, current_end = sorted_starts[0], found_positions[sorted_starts[0]]
        for start in sorted_starts[1:]:
            if start <= (current_end + 10):
                current_end = max(current_end, found_positions[start])
            else:
                combined_positions.append((current_start, current_end))
                current_start, current_end = start, found_positions[start]
        combined_positions.append((current_start, current_end))

    # Construct pos_tokens
    pos_tokens = []
    prev_end = 0
    for start, end in combined_positions:
        if end-start > 15: # Only combine if there is a significant amount of matched text. Avoids picking up single words like 'and' etc.
            pos_tokens.append(full_text[prev_end:start])
            pos_tokens.append('<mark style="color:black;">' + full_text[start:end] + '</mark>')
            prev_end = end
    pos_tokens.append(full_text[prev_end:])

    return "".join(pos_tokens)


# # Chat history functions

def clear_chat(chat_history_state, sources, chat_message, current_topic):
    chat_history_state = []
    sources = ''
    chat_message = ''
    current_topic = ''

    return chat_history_state, sources, chat_message, current_topic


# Keyword functions

def remove_q_stopwords(question): # Remove stopwords from question. Not used at the moment 
    # Prepare keywords from question by removing stopwords
    text = question.lower()

    # Remove numbers
    text = re.sub('[0-9]', '', text)

    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    #text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in tokens_without_sw:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)
     


    new_question_keywords = ' '.join(result)
    return new_question_keywords

def remove_q_ner_extractor(question):
    
    predict_out = ner_model.predict(question)



    predict_tokens = [' '.join(v for k, v in d.items() if k == 'span') for d in predict_out]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in predict_tokens:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)
     


    new_question_keywords = ' '.join(result).lower()
    return new_question_keywords

def apply_lemmatize(text, wnl=WordNetLemmatizer()):

    def prep_for_lemma(text):

        # Remove numbers
        text = re.sub('[0-9]', '', text)
        print(text)

        tokenizer = RegexpTokenizer(r'\w+')
        text_tokens = tokenizer.tokenize(text)
        #text_tokens = word_tokenize(text)

        return text_tokens

    tokens = prep_for_lemma(text)

    def lem_word(word):
    
        if len(word) > 3: out_word = wnl.lemmatize(word)
        else: out_word = word

        return out_word

    return [lem_word(token) for token in tokens]

def keybert_keywords(text, n, kw_model):
    tokens_lemma = apply_lemmatize(text)
    lemmatised_text = ' '.join(tokens_lemma)

    keywords_text = KeyBERT(model=kw_model).extract_keywords(lemmatised_text, stop_words='english', top_n=n, 
                                                   keyphrase_ngram_range=(1, 1))
    keywords_list = [item[0] for item in keywords_text]

    return keywords_list
    
# Gradio functions
def turn_off_interactivity(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

def restore_interactivity():
        return gr.update(interactive=True)

def update_message(dropdown_value):
        return gr.Textbox.update(value=dropdown_value)

def hide_block():
        return gr.Radio.update(visible=False)
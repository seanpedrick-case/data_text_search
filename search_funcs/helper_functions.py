import os
import re
import pandas as pd
import gradio as gr
import os
import shutil
import getpass
import gzip
import pickle
import numpy as np

# Attempt to delete content of gradio temp folder
def get_temp_folder_path():
    username = getpass.getuser()
    return os.path.join('C:\\Users', username, 'AppData\\Local\\Temp\\gradio')

def empty_folder(directory_path):
    if not os.path.exists(directory_path):
        #print(f"The directory {directory_path} does not exist. No temporary files from previous app use found to delete.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            #print(f'Failed to delete {file_path}. Reason: {e}')
            print('')



def get_file_path_end(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def get_file_path_end_with_ext(file_path):
    match = re.search(r'(.*[\/\\])?(.+)$', file_path)
        
    filename_end = match.group(2) if match else ''

    return filename_end

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.pkl.gz'):
        return 'pkl.gz'
    #elif filename.endswith('.gz'):
    #    return 'gz'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
        
    print("Loading in file")

    if file_type == 'csv':
        file = pd.read_csv(filename, low_memory=False).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'xlsx':
        file = pd.read_excel(filename).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'parquet':
        file = pd.read_parquet(filename).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'pkl.gz':
        with gzip.open(filename, 'rb') as file:
            file = pickle.load(file)
    #elif file_type == ".gz":
    #    with gzip.open(filename, 'rb') as file:
    #        file = pickle.load(file)

    print("File load complete")

    return file

def put_columns_in_df(in_file, in_bm25_column):
    '''
    When file is loaded, update the column dropdown choices
    '''
    new_choices = []
    concat_choices = []
    index_load = None
    embed_load = np.array([])
    out_message = ""

    file_list = [string.name for string in in_file]

    #print(file_list)

    data_file_names = [string.lower() for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower() and "search_index" not in string.lower()]

    if not data_file_names:
        out_message = "Please load in at least one csv/Excel/parquet data file."
        print(out_message)
        return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), pd.DataFrame(), bm25_load, out_message

    data_file_name = data_file_names[0]
   
    
    df = read_file(data_file_name)

    if "pkl" not in data_file_name:

        new_choices = list(df.columns)

    elif "search_index" in data_file_name:
        # If only the search_index found, need a data file too
        new_choices = []

    else: new_choices = ["page_contents"] + list(df[0].metadata.keys()) #["Documents"]
    #print(new_choices)

    concat_choices.extend(new_choices)

    # Check if there is a search index file already
    index_file_names = [string.lower() for string in file_list if "gz" in string.lower()]

    if index_file_names:
        index_file_name = index_file_names[0]
        index_load = read_file(index_file_name)

    embeddings_file_names = [string.lower() for string in file_list if "embedding" in string.lower()]

    if embeddings_file_names:
        print("Loading embeddings from file.")
        embed_load = np.load(embeddings_file_names[0])['arr_0']

        # If embedding files have 'super_compress' in the title, they have been multiplied by 100 before save
        if "compress" in embeddings_file_names[0]:
            embed_load /= 100
    else:
        embed_load = np.array([])

    out_message = "Initial data check successful. Next, choose a data column to search in the drop down above, then click 'Load data'"
    print(out_message)
        
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, index_load, embed_load, out_message

def put_columns_in_join_df(in_file):
    '''
    When file is loaded, update the column dropdown choices
    '''
    new_df = pd.DataFrame()
    #print("in_bm25_column")

    new_choices = []
    concat_choices = []
    
    
    new_df = read_file(in_file.name)
    new_choices = list(new_df.columns)

    #print(new_choices)

    concat_choices.extend(new_choices)

    out_message = "File load successful. Now select a column to join below."    
        
    return gr.Dropdown(choices=concat_choices), new_df, out_message

def dummy_function(gradio_component):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None    

def display_info(info_component):
    gr.Info(info_component)


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

# Openpyxl functions for output
from openpyxl import Workbook
from openpyxl.cell.text import InlineFont 
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl.styles import Font, Alignment

megabyte = 1024 * 1024  # Bytes in a megabyte
file_size_mb = 500  # Size in megabytes
file_size_bytes_500mb =  megabyte * file_size_mb

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
        file = pd.read_csv(filename, low_memory=False).reset_index()#.drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'xlsx':
        file = pd.read_excel(filename).reset_index()#.drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'parquet':
        file = pd.read_parquet(filename).reset_index()#.drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'pkl.gz':
        with gzip.open(filename, 'rb') as file:
            file = pickle.load(file)
    #elif file_type == ".gz":
    #    with gzip.open(filename, 'rb') as file:
    #        file = pickle.load(file)

    print("File load complete")

    return file

def initial_data_load(in_file):
    '''
    When file is loaded, update the column dropdown choices
    '''
    new_choices = []
    concat_choices = []
    index_load = None
    embed_load = np.array([])
    tokenised_load =[]
    out_message = ""
    current_source = ""
    df = pd.DataFrame()

    file_list = [string.name for string in in_file]

    #print(file_list)

    data_file_names = [string for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower() and "search_index" not in string.lower()]
    print(data_file_names)

    if not data_file_names:
        out_message = "Please load in at least one csv/Excel/parquet data file."
        print(out_message)
        return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), pd.DataFrame(), pd.DataFrame(), index_load, embed_load, tokenised_load, out_message, None

    # This if you have loaded in a documents object for the semantic search
    if "pkl" in data_file_names[0]: 
        df = read_file(data_file_names[0])
        new_choices = list(df[0].metadata.keys()) #["Documents"] #["page_contents"] + 
        current_source = get_file_path_end_with_ext(data_file_names[0])

    # This if you have loaded in a csv/parquets/xlsx
    else:
        for file in data_file_names:

            current_source = current_source + get_file_path_end_with_ext(file) + " "
        
            # Get the size of the file
            print("Checking file size")
            file_size = os.path.getsize(file)
            if file_size > file_size_bytes_500mb:
                out_message = "Data file greater than 500mb in size. Please use smaller sizes."
                print(out_message)
                return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), pd.DataFrame(), pd.DataFrame(), index_load, embed_load, tokenised_load, out_message, None


            df_new = read_file(file)

            df = pd.concat([df, df_new], ignore_index = True)

        new_choices = list(df.columns)

    concat_choices.extend(new_choices)

    # Check if there is a search index file already
    index_file_names = [string for string in file_list if "gz" in string.lower()]

    if index_file_names:
        index_file_name = index_file_names[0]
        index_load = read_file(index_file_name)

    embeddings_file_names = [string for string in file_list if "embedding" in string.lower()]

    if embeddings_file_names:
        print("Loading embeddings from file.")
        embed_load = np.load(embeddings_file_names[0])['arr_0']

        # If embedding files have 'super_compress' in the title, they have been multiplied by 100 before save
        if "compress" in embeddings_file_names[0]:
            embed_load /= 100
    else:
        embed_load = np.array([])

    tokenised_file_names = [string for string in file_list if "tokenised" in string.lower()]
    if tokenised_file_names:
        tokenised_load = read_file(tokenised_file_names[0])

    out_message = "Initial data check successful. Next, choose a data column to search in the drop down above, then click 'Load data'"
    print(out_message)
        
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, df, index_load, embed_load, tokenised_load, out_message, current_source

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


    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None    

def display_info(info_component):
    gr.Info(info_component)

def highlight_found_text(search_text: str, full_text: str) -> str:
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
            return text
        elif isinstance(text, list):
            return text[i][0]
        else:
            return ""

    def extract_search_text_from_input(text):
        if isinstance(text, str):
            return text
        elif isinstance(text, list):
            return text[-1][1]
        else:
            return ""

    full_text = extract_text_from_input(full_text)
    search_text = extract_search_text_from_input(search_text)

    sections = search_text.split(sep = " ")

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
        if end-start > 1: # Only combine if there is a significant amount of matched text. Avoids picking up single words like 'and' etc.
            pos_tokens.append(full_text[prev_end:start])
            pos_tokens.append('<mark style="color:black;">' + full_text[start:end] + '</mark>')
            prev_end = end
    pos_tokens.append(full_text[prev_end:])

    return "".join(pos_tokens), combined_positions

def create_rich_text_cell_from_positions(full_text, combined_positions):
    # Construct pos_tokens
    red = InlineFont(color='00FF0000')
    rich_text_cell = CellRichText()

    prev_end = 0
    for start, end in combined_positions:
        if end-start > 1: # Only combine if there is a significant amount of matched text. Avoids picking up single words like 'and' etc.
            rich_text_cell.append(full_text[prev_end:start])
            rich_text_cell.append(TextBlock(red, full_text[start:end]))
            prev_end = end
    rich_text_cell.append(full_text[prev_end:])

    return rich_text_cell

def create_highlighted_excel_wb(df, search_text, column_to_highlight):

    # Create a new Excel workbook
    wb = Workbook()
    sheet = wb.active

    # Insert headers into the worksheet, make bold
    sheet.append(df.columns.tolist())
    for cell in sheet[1]:
        cell.font = Font(bold=True)

    column_width = 150  # Adjust as needed
    relevant_column_no = (df.columns == column_to_highlight).argmax() + 1
    print(relevant_column_no)
    sheet.column_dimensions[sheet.cell(row=1, column=relevant_column_no).column_letter].width = column_width

    # Find substrings in cells and highlight
    for r_idx, row in enumerate(df.itertuples(), start=2):
        for c_idx, cell_value in enumerate(row[1:], start=1):
            sheet.cell(row=r_idx, column=c_idx, value=cell_value)
            if df.columns[c_idx - 1] == column_to_highlight:

                html_text, combined_positions = highlight_found_text(search_text, cell_value)
                sheet.cell(row=r_idx, column=c_idx).value = create_rich_text_cell_from_positions(cell_value, combined_positions)
                sheet.cell(row=r_idx, column=c_idx).alignment = Alignment(wrap_text=True)

    return wb
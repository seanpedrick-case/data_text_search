import time
import ast
import gzip
import pandas as pd
import gradio as gr
import pickle
from typing import Type, List, Literal
from pydantic import BaseModel, Field

# Creating an alias for pandas DataFrame using Type
PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]

class Document(BaseModel):
    """Class for storing a piece of text and associated metadata. Implementation adapted from Langchain code: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py"""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"

from search_funcs.helper_functions import get_file_path_end, ensure_output_folder_exists
from search_funcs.bm25_functions import save_prepared_bm25_data, output_folder
from search_funcs.clean_funcs import initial_clean

def combine_metadata_columns(df:PandasDataFrame, cols:List[str]) -> PandasSeries:
    '''
    Construct a metadata column as a string version of a dictionary for later parsing.

    Parameters:
    - df (PandasDataFrame): Data frame of search data.
	- cols (List[str]): List of column names that will be included in the output metadata column.

	Returns:
	- PandasSeries: A series containing the metadata elements combined into a dictionary format as a string.
    '''

    df['metadata'] = '{'
    df['blank_column'] = ''

    for n, col in enumerate(cols):
        df[col] = df[col].astype(str).str.replace('"',"'").str.replace('\n', ' ').str.replace('\r', ' ').str.replace('\r\n', ' ').str.cat(df['blank_column'].astype(str), sep="")

        df['metadata'] = df['metadata'] + '"' + cols[n] + '": "' + df[col] + '", '


    df['metadata'] = (df['metadata'] + "}").str.replace(', }', '}').str.replace('", }"', '}')

    return df['metadata']

def clean_line_breaks(text:str):
    '''Replace \n and \r\n with a space'''
    return text.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')

def parse_metadata(row):
    '''
    Parse a string instance of a dictionary into a Python object.
    '''
    try:
        # Ensure the 'title' field is a string and clean line breaks
        #if 'TITLE' in row:
        #    row['TITLE'] = clean_line_breaks(row['TITLE'])

        # Convert the row to a string if it's not already
        row_str = str(row) if not isinstance(row, str) else row

        row_str.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')

        # Parse the string
        metadata = ast.literal_eval(row_str)
        # Process metadata
        return metadata
    except SyntaxError as e:
        print(f"Failed to parse metadata: {row_str}")
        print(f"Error: {e}")
        # Handle the error or log it
        return None  # or some default value

def csv_excel_text_to_docs(df:PandasDataFrame, in_file:List[str], text_column:str, clean:str = "No", return_intermediate_files:str = "No", progress=gr.Progress(track_tqdm=True)) -> tuple:
    """Converts a DataFrame's content to a list of dictionaries in the 'Document' format, containing page_content and associated metadata.
    
    Parameters:
    - df (PandasDataFrame): Data frame of search data.
	- in_file (List[str]): List of input file names.
	- text_column (str): The text column that will be searched.
	- clean (str): Whether the text is cleaned before searching.
	- return_intermediate_files (str): Whether intermediate processing files are saved to file.
	- progress (gr.Progress, optional): The progress tracker for the operation.

	Returns:
	- tuple: A tuple containing data outputs in a Document class format, an output message, and a list of output file paths.
    """

    ensure_output_folder_exists(output_folder)
    output_list = []

    if not in_file:
        return None, "Please load in at least one file.", output_list

    progress(0, desc = "Loading in data")
    
    file_list = [string.name for string in in_file]

    data_file_names = [string for string in file_list if "tokenised" not in string and "npz" not in string.lower()]

    if not data_file_names:
        return doc_sections, "Please load in at least one csv/Excel/parquet data file.", output_list

    if not text_column:
        return None, "Please enter a column name to search", output_list

    data_file_name = data_file_names[0]

    # Check if file is a document format, and explode out as needed
    if "prepared_docs" in data_file_name:
        print("Loading in documents from file.")

        doc_sections = df

        # Convert each element in the Series to a Document instance

        return doc_sections, "Finished preparing documents", output_list

    ingest_tic = time.perf_counter()

    doc_sections = []
    df[text_column] = df[text_column].astype(str).str.strip() # Ensure column is a string column

    original_text_column = text_column

    if clean == "Yes":
        progress(0.1, desc = "Cleaning data")
        clean_tic = time.perf_counter()
        print("Starting data clean.")
        
        df_list = list(df[text_column])
        df_list = initial_clean(df_list)

        # Save to file if you have cleaned the data. Text column has now been renamed with '_cleaned' at the send
        out_file_name, text_column, df = save_prepared_bm25_data(data_file_name, df_list, df, text_column)

        df[text_column] = df_list

        clean_toc = time.perf_counter()
        clean_time_out = f"Cleaning the text took {clean_toc - clean_tic:0.1f} seconds."
        print(clean_time_out)

    cols = [col for col in df.columns if col != original_text_column]

    df["metadata"] = combine_metadata_columns(df, cols)

    progress(0.3, desc = "Converting data to document format")

    # Create a list of Document objects
    doc_sections = [Document(page_content=row[text_column], 
                        metadata= parse_metadata(row["metadata"]))
                for index, row in progress.tqdm(df.iterrows(), desc = "Splitting up text", unit = "rows")]

    ingest_toc = time.perf_counter()

    ingest_time_out = f"Preparing documents took {ingest_toc - ingest_tic:0.1f} seconds"
    print(ingest_time_out)

    if return_intermediate_files == "Yes":
        progress(0.5, desc = "Saving prepared documents")
        data_file_out_name_no_ext = get_file_path_end(data_file_name)
        file_name = data_file_out_name_no_ext

        if clean == "No":
            out_doc_file_name = output_folder + file_name + "_prepared_docs.pkl.gz"
            with gzip.open(out_doc_file_name, 'wb') as file:
                pickle.dump(doc_sections, file)

        elif clean == "Yes":
            out_doc_file_name = output_folder + file_name + "_cleaned_prepared_docs.pkl.gz"
            with gzip.open(out_doc_file_name, 'wb') as file:
                pickle.dump(doc_sections, file)

        output_list.append(out_doc_file_name)
        print("Documents saved to file.")

    return doc_sections, "Finished preparing documents.", output_list
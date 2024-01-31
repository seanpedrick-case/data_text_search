# Install/ import stuff we need

import os
import time
import re
import ast
import gzip
import pandas as pd
import gradio as gr
from typing import Type, List, Literal
#from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field

# Creating an alias for pandas DataFrame using Type
PandasDataFrame = Type[pd.DataFrame]

class Document(BaseModel):
    """Class for storing a piece of text and associated metadata. Implementation adapted from Langchain code: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py"""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"

# Constants for chunking - not currently used
split_strat = ["\n\n", "\n", ". ", "! ", "? "]
chunk_size = 512
chunk_overlap = 0
start_index = True

from search_funcs.helper_functions import get_file_path_end_with_ext, detect_file_type, get_file_path_end
from search_funcs.bm25_functions import save_prepared_bm25_data
from search_funcs.clean_funcs import initial_clean

## Parse files
# def detect_file_type(file_path):
#         """
#         Determine the file type based on its extension.
    
#         Parameters:
#             file_path (str): Path to the file.
    
#         Returns:
#             str: File extension (e.g., '.pdf', '.docx', '.txt', '.html').
#         """
#         return os.path.splitext(file_path)[1].lower()

def parse_file_not_used(file_paths, text_column='text'):
    """
    Accepts a list of file paths, determines each file's type based on its extension,
    and passes it to the relevant parsing function.
    
    Parameters:
        file_paths (list): List of file paths.
        text_column (str): Name of the column in CSV/Excel files that contains the text content.
    
    Returns:
        dict: A dictionary with file paths as keys and their parsed content (or error message) as values.
    """
    
    

    if not isinstance(file_paths, list):
        raise ValueError("Expected a list of file paths.")
    
    extension_to_parser = {
        # '.pdf': parse_pdf,
        # '.docx': parse_docx,
        # '.txt': parse_txt,
        # '.html': parse_html,
        # '.htm': parse_html,  # Considering both .html and .htm for HTML files
        '.csv': lambda file_path: parse_csv_or_excel(file_path, text_column),
        '.xlsx': lambda file_path: parse_csv_or_excel(file_path, text_column),
        '.parquet': lambda file_path: parse_csv_or_excel(file_path, text_column)
    }
    
    parsed_contents = {}
    file_names = []

    for file_path in file_paths:
        #print(file_path.name)
        #file = open(file_path.name, 'r')
        #print(file)
        file_extension = detect_file_type(file_path.name)
        if file_extension in extension_to_parser:
            parsed_contents[file_path.name] = extension_to_parser[file_extension](file_path.name)
        else:
            parsed_contents[file_path.name] = f"Unsupported file type: {file_extension}"

        filename_end = get_file_path_end_with_ext(file_path.name)

        file_names.append(filename_end)
    
    return parsed_contents, file_names

def text_regex_clean(text):
    # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # If a double newline ends in a letter, add a full stop.
        text = re.sub(r'(?<=[a-zA-Z])\n\n', '.\n\n', text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"  ", " ", text)
        # Add full stops and new lines between words with no space between where the second one has a capital letter
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', '. \n\n', text)

        return text

def parse_csv_or_excel(file_path, data_state, text_column = "text"):
        """
        Read in a CSV or Excel file.
        
        Parameters:
            file_path (str): Path to the CSV file.
            text_column (str): Name of the column in the CSV file that contains the text content.
        
        Returns:
            Pandas DataFrame: Dataframe output from file read
        """

        #out_df = pd.DataFrame()

        file_list = [string.name for string in file_path]

        #print(file_list)

        data_file_names = [string.lower() for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower()]# and "gz" not in string.lower()]
        
        data_file_name = data_file_names[0]
        
        #for file_path in file_paths:
        file_name = get_file_path_end_with_ext(data_file_name)

        #print(file_extension)

        # if file_extension == "csv":
        #         df = pd.read_csv(data_file_names[0], low_memory=False)
        #         if text_column not in df.columns: return pd.DataFrame(), ['Please choose a valid column name']
        #         df['source'] = file_name
        #         df['page_section'] = ""
        # elif file_extension == "xlsx":
        #         df = pd.read_excel(data_file_names[0], engine='openpyxl')
        #         if text_column not in df.columns: return pd.DataFrame(), ['Please choose a valid column name']
        #         df['source'] = file_name
        #         df['page_section'] = ""
        # elif file_extension == "parquet":
        #         df = pd.read_parquet(data_file_names[0])
        #         if text_column not in df.columns: return pd.DataFrame(), ['Please choose a valid column name']
        #         df['source'] = file_name
        #         df['page_section'] = ""
        # else:
        #         print(f"Unsupported file type: {file_extension}")
        #         return pd.DataFrame(), ['Please choose a valid file type']
        
        df = data_state
        #df['source'] = file_name
        #df['page_section'] = ""

        message = "Loaded in file. Now converting to document format."
        print(message)

        return df, file_name, message


# +
# Convert parsed text to docs
# -

def write_out_metadata_as_string(metadata_in):
    # If metadata_in is a single dictionary, wrap it in a list
    if isinstance(metadata_in, dict):
        metadata_in = [metadata_in]

    metadata_string = [f"{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}" for d in metadata_in] # ['metadata']
    return metadata_string

def combine_metadata_columns(df, cols):

    df['metadata'] = '{'
    df['blank_column'] = ''

    for n, col in enumerate(cols):
        df[col] = df[col].astype(str).str.replace('"',"'").str.replace('\n', ' ').str.replace('\r', ' ').str.replace('\r\n', ' ').str.cat(df['blank_column'].astype(str), sep="")

        df['metadata'] = df['metadata'] + '"' + cols[n] + '": "' + df[col] + '", '


    df['metadata'] = (df['metadata'] + "}").str.replace(', }', '}').str.replace('", }"', '}')

    return df['metadata']

def split_string_into_chunks(input_string, max_length, split_symbols):
    # Check if input_string or split_symbols are empty
    if not input_string or not split_symbols:
        return [input_string]

    chunks = []
    current_chunk = ""
    
    for char in input_string:
        current_chunk += char
        if len(current_chunk) >= max_length or char in split_symbols:
            # Add the current chunk to the chunks list
            chunks.append(current_chunk)
            current_chunk = ""
    
    # Adding any remaining part of the string
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def clean_line_breaks(text):
    # Replace \n and \r\n with a space
    return text.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')

def parse_metadata(row):
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

# def csv_excel_text_to_docs_deprecated(df, text_column='text', chunk_size=None) -> List[Document]:
#     """Converts a DataFrame's content to a list of Documents with metadata."""

#     print("Converting to documents.")

#     doc_sections = []
#     df[text_column] = df[text_column].astype(str) # Ensure column is a string column

#     # For each row in the dataframe
#     for idx, row in df.iterrows():
#         # Extract the text content for the document
#         doc_content = row[text_column]
        
#         # Generate metadata containing other columns' data
#         metadata = {"row": idx + 1}
#         for col, value in row.items():
#             if col != text_column:
#                 metadata[col] = value

#         metadata_string = write_out_metadata_as_string(metadata)[0]      

#         # If chunk_size is provided, split the text into chunks
#         if chunk_size:
#             sections = split_string_into_chunks(doc_content, chunk_size, split_strat)

#             # Langchain usage deprecated
#             # text_splitter = RecursiveCharacterTextSplitter(
#             #    chunk_size=chunk_size,
#             #    chunk_overlap=chunk_overlap,
#             #    split_strat=split_strat,
#             #    start_index=start_index                
#             # ) #Other arguments as required by the splitter

#             # sections = text_splitter.split_text(doc_content)
   
#             # For each section, create a Document object
#             for i, section in enumerate(sections):
#                 section = '. '.join([metadata_string, section])
#                 doc = Document(page_content=section, 
#                               metadata={**metadata, "section": i, "row_section": f"{metadata['row']}-{i}"})
#                 doc_sections.append(doc)

#         else:
#             # If no chunk_size is provided, create a single Document object for the row
#             #doc_content = '. '.join([metadata_string, doc_content])
#             doc = Document(page_content=doc_content, metadata=metadata)
#             doc_sections.append(doc)

#         message = "Data converted to document format. Now creating/loading document embeddings."
#         print(message)

#     return doc_sections, message

def csv_excel_text_to_docs(df, in_file, text_column, clean = "No", return_intermediate_files = "No", chunk_size=None, progress=gr.Progress(track_tqdm=True)) -> List[Document]:
    """Converts a DataFrame's content to a list of dictionaries in the 'Document' format, containing page_content and associated metadata."""
    if not in_file:
        return None, "Please load in at least one file.", data_state, None, None, None

    progress(0, desc = "Loading in data")
    
    file_list = [string.name for string in in_file]

    data_file_names = [string.lower() for string in file_list if "tokenised" not in string and "npz" not in string.lower()]

    if not data_file_names:
        return doc_sections, "Please load in at least one csv/Excel/parquet data file."

    if not text_column:
        return None, "Please enter a column name to search", data_state, None, None, None

    data_file_name = data_file_names[0]

    # Check if file is a document format, and explode out as needed
    if "prepared_docs" in data_file_name:
        print("Loading in documents from file.")

        #print(df[0:5])
        #section_series = df.iloc[:,0]
        #section_series = "{" + section_series + "}"

        doc_sections = df

        #print(doc_sections[0])

        # Convert each element in the Series to a Document instance
        #doc_sections = section_series.apply(lambda x: Document(**x))

        return doc_sections, "Finished preparing documents"
    #    df = document_to_dataframe(df.iloc[:,0])

    ingest_tic = time.perf_counter()

    doc_sections = []
    df[text_column] = df[text_column].astype(str).str.strip() # Ensure column is a string column

    if clean == "Yes":
        progress(0.1, desc = "Cleaning data")
        clean_tic = time.perf_counter()
        print("Starting data clean.")
        
        #df = df.drop_duplicates(text_column)
        
        df[text_column] = initial_clean(df[text_column])
        df_list = list(df[text_column])

        # Save to file if you have cleaned the data
        out_file_name, text_column = save_prepared_bm25_data(data_file_name, df_list, df, text_column)

        clean_toc = time.perf_counter()
        clean_time_out = f"Cleaning the text took {clean_toc - clean_tic:0.1f} seconds."
        print(clean_time_out)

    cols = [col for col in df.columns if col != text_column]

    df["metadata"] = combine_metadata_columns(df, cols)

    df = df.rename(columns={text_column:"page_content"})

    #print(df[["page_content", "metadata"]].to_dict(orient='records'))

    #doc_sections = df[["page_content", "metadata"]].to_dict(orient='records')
    #doc_sections = [Document(**row) for row in df[["page_content", "metadata"]].to_dict(orient='records')]

    progress(0.3, desc = "Converting data to document format")

    # Create a list of Document objects
    doc_sections = [Document(page_content=row['page_content'], 
                        metadata= parse_metadata(row["metadata"]))
                for index, row in progress.tqdm(df.iterrows(), desc = "Splitting up text", unit = "rows")]

    ingest_toc = time.perf_counter()

    ingest_time_out = f"Preparing documents took {ingest_toc - ingest_tic:0.1f} seconds"
    print(ingest_time_out)

    if return_intermediate_files == "Yes":
        progress(0.5, desc = "Saving prepared documents")
        data_file_out_name_no_ext = get_file_path_end(data_file_name)
        file_name = data_file_out_name_no_ext
        #print(doc_sections)
        #page_content_series_string = pd.Series(doc_sections).astype(str)
        #page_content_series_string = page_content_series_string.str.replace(" type='Document'", "").str.replace("' metadata=", "', 'metadata':").str.replace("page_content=", "{'page_content':")
        #page_content_series_string = page_content_series_string + "}"
        #print(page_content_series_string[0])
        #metadata_series_string = pd.Series(doc_sections[1]).astype(str)

        import pickle

        if clean == "No":
            #pd.DataFrame(data = {"Documents":page_content_series_string}).to_parquet(file_name + "_prepared_docs.parquet")

            with gzip.open(file_name + "_prepared_docs.pkl.gz", 'wb') as file:
                pickle.dump(doc_sections, file)

            #pd.Series(doc_sections).to_pickle(file_name + "_prepared_docs.pkl")
        elif clean == "Yes":
            #pd.DataFrame(data = {"Documents":page_content_series_string}).to_parquet(file_name + "_prepared_docs_clean.parquet")

            with gzip.open(file_name + "_prepared_docs_clean.pkl.gz", 'wb') as file:
                pickle.dump(doc_sections, file)

            #pd.Series(doc_sections).to_pickle(file_name + "_prepared_docs_clean.pkl")
        print("Documents saved to file.")

    return doc_sections, "Finished preparing documents."


def document_to_dataframe(documents):
    '''
    Convert an object in document format to pandas dataframe
    '''
    rows = []

    for doc in documents:
        # Convert Document to dictionary and extract metadata
        doc_dict = doc.dict()
        metadata = doc_dict.pop('metadata')

        # Add the page_content and type to the metadata
        metadata['page_content'] = doc_dict['page_content']
        metadata['type'] = doc_dict['type']

        # Add to the list of rows
        rows.append(metadata)

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows)
    return df

# Example usage
#documents = [
#    Document(page_content="Example content 1", metadata={"author": "Author 1", "year": 2021}),
#    Document(page_content="Example content 2", metadata={"author": "Author 2", "year": 2022})
#]

#df = document_to_dataframe(documents)
#df
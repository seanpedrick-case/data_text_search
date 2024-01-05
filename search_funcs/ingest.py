# Install/ import stuff we need

import os
import time
import re
import ast
import pandas as pd
import gradio as gr
from typing import Type, List, Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field

# Creating an alias for pandas DataFrame using Type
PandasDataFrame = Type[pd.DataFrame]

# class Document(BaseModel):
#     """Class for storing a piece of text and associated metadata. Implementation adapted from Langchain code: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py"""

#     page_content: str
#     """String text."""
#     metadata: dict = Field(default_factory=dict)
#     """Arbitrary metadata about the page content (e.g., source, relationships to other
#         documents, etc.).
#     """
#     type: Literal["Document"] = "Document"

class Document(BaseModel):
    """Class for storing a piece of text and associated metadata. Implementation adapted from Langchain code: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py"""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"


# -

split_strat = ["\n\n", "\n", ". ", "! ", "? "]
chunk_size = 500
chunk_overlap = 0
start_index = True

## Parse files
def determine_file_type(file_path):
        """
        Determine the file type based on its extension.
    
        Parameters:
            file_path (str): Path to the file.
    
        Returns:
            str: File extension (e.g., '.pdf', '.docx', '.txt', '.html').
        """
        return os.path.splitext(file_path)[1].lower()

def parse_file(file_paths, text_column='text'):
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
        print(file_path.name)
        #file = open(file_path.name, 'r')
        #print(file)
        file_extension = determine_file_type(file_path.name)
        if file_extension in extension_to_parser:
            parsed_contents[file_path.name] = extension_to_parser[file_extension](file_path.name)
        else:
            parsed_contents[file_path.name] = f"Unsupported file type: {file_extension}"

        filename_end = get_file_path_end(file_path.name)

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

def parse_csv_or_excel(file_path, text_column = "text"):
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

        print(file_list)

        data_file_names = [string for string in file_list if "tokenised" not in string]
        
        
        #for file_path in file_paths:
        file_extension = determine_file_type(data_file_names[0])
        file_name = get_file_path_end(data_file_names[0])
        file_names = [file_name]

        print(file_extension)

        if file_extension == ".csv":
                df = pd.read_csv(data_file_names[0], low_memory=False)
                if text_column not in df.columns: return pd.DataFrame(), ['Please choose a valid column name']
                df['source'] = file_name
                df['page_section'] = ""
        elif file_extension == ".xlsx":
                df = pd.read_excel(data_file_names[0], engine='openpyxl')
                if text_column not in df.columns: return pd.DataFrame(), ['Please choose a valid column name']
                df['source'] = file_name
                df['page_section'] = ""
        elif file_extension == ".parquet":
                df = pd.read_parquet(data_file_names[0])
                if text_column not in df.columns: return pd.DataFrame(), ['Please choose a valid column name']
                df['source'] = file_name
                df['page_section'] = ""
        else:
                print(f"Unsupported file type: {file_extension}")
                return pd.DataFrame(), ['Please choose a valid file type']
        
        message = "Loaded in file. Now converting to document format."
        print(message)

        return df, file_names, message

def get_file_path_end(file_path):
    match = re.search(r'(.*[\/\\])?(.+)$', file_path)
        
    filename_end = match.group(2) if match else ''

    return filename_end

# +
# Convert parsed text to docs
# -

def text_to_docs(text_dict: dict, chunk_size: int = chunk_size) -> List[Document]:
    """
    Converts the output of parse_file (a dictionary of file paths to content)
    to a list of Documents with metadata.
    """
    
    doc_sections = []
    parent_doc_sections = []

    for file_path, content in text_dict.items():
        ext = os.path.splitext(file_path)[1].lower()

        # Depending on the file extension, handle the content
        # if ext == '.pdf':
        #     docs, page_docs = pdf_text_to_docs(content, chunk_size)
        # elif ext in ['.html', '.htm', '.txt', '.docx']:
        #     docs = html_text_to_docs(content, chunk_size)
        if ext in ['.csv', '.xlsx']:
            docs, page_docs = csv_excel_text_to_docs(content, chunk_size)
        else:
            print(f"Unsupported file type {ext} for {file_path}. Skipping.")
            continue

        
        filename_end = get_file_path_end(file_path)

        #match = re.search(r'(.*[\/\\])?(.+)$', file_path)
        #filename_end = match.group(2) if match else ''

        # Add filename as metadata
        for doc in docs: doc.metadata["source"] = filename_end
        #for parent_doc in parent_docs: parent_doc.metadata["source"] = filename_end
        
        doc_sections.extend(docs)
        #parent_doc_sections.extend(parent_docs)

    return doc_sections#, page_docs


def write_out_metadata_as_string(metadata_in):
    # If metadata_in is a single dictionary, wrap it in a list
    if isinstance(metadata_in, dict):
        metadata_in = [metadata_in]

    metadata_string = [f"{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}" for d in metadata_in] # ['metadata']
    return metadata_string

def combine_metadata_columns(df, cols):

    df['metadatas'] = "{"
    df['blank_column'] = ""

    for n, col in enumerate(cols):
        df[col] = df[col].astype(str).str.replace('"',"'").str.replace('\n', ' ').str.replace('\r', ' ').str.replace('\r\n', ' ').str.cat(df['blank_column'].astype(str), sep="")

        df['metadatas'] = df['metadatas'] + '"' + cols[n] + '": "' + df[col] + '", '


    df['metadatas'] = (df['metadatas'] + "}").str.replace(', }', '}')

    return df['metadatas']

def csv_excel_text_to_docs(df, text_column='text', chunk_size=None) -> List[Document]:
    """Converts a DataFrame's content to a list of Documents with metadata."""
    
    #print(df.head())

    print("Converting to documents.")

    doc_sections = []
    df[text_column] = df[text_column].astype(str) # Ensure column is a string column

    # For each row in the dataframe
    for idx, row in df.iterrows():
        # Extract the text content for the document
        doc_content = row[text_column]
        
        # Generate metadata containing other columns' data
        metadata = {"row": idx + 1}
        for col, value in row.items():
            if col != text_column:
                metadata[col] = value

        metadata_string = write_out_metadata_as_string(metadata)[0]      

        # If chunk_size is provided, split the text into chunks
        if chunk_size:
            # Assuming you have a text splitter function similar to the PDF handling
            text_splitter = RecursiveCharacterTextSplitter(
               chunk_size=chunk_size,
               chunk_overlap=chunk_overlap,
               split_strat=split_strat,
               start_index=start_index                
            ) #Other arguments as required by the splitter

            sections = text_splitter.split_text(doc_content)

            
            # For each section, create a Document object
            for i, section in enumerate(sections):
                section = '. '.join([metadata_string, section])
                doc = Document(page_content=section, 
                              metadata={**metadata, "section": i, "row_section": f"{metadata['row']}-{i}"})
                doc_sections.append(doc)
            
            #print("Chunking currently disabled")

        else:
            # If no chunk_size is provided, create a single Document object for the row
            #doc_content = '. '.join([metadata_string, doc_content])
            doc = Document(page_content=doc_content, metadata=metadata)
            doc_sections.append(doc)

        message = "Data converted to document format. Now creating/loading document embeddings."
        print(message)

    return doc_sections, message



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

def csv_excel_text_to_docs(df, text_column='text', chunk_size=None, progress=gr.Progress()) -> List[Document]:
    """Converts a DataFrame's content to a list of dictionaries in the 'Document' format, containing page_content and associated metadata."""
    
    ingest_tic = time.perf_counter()

    doc_sections = []
    df[text_column] = df[text_column].astype(str).str.strip() # Ensure column is a string column

    cols = [col for col in df.columns if col != text_column]

    df["metadata"] = combine_metadata_columns(df, cols)

    df = df.rename(columns={text_column:"page_content"})

    #print(df[["page_content", "metadata"]].to_dict(orient='records'))

    #doc_sections = df[["page_content", "metadata"]].to_dict(orient='records')
    #doc_sections = [Document(**row) for row in df[["page_content", "metadata"]].to_dict(orient='records')]

    # Create a list of Document objects
    doc_sections = [Document(page_content=row['page_content'], 
                        metadata= parse_metadata(row["metadata"]))
               for index, row in progress.tqdm(df.iterrows(), desc = "Splitting up text", unit = "rows")]
    
    ingest_toc = time.perf_counter()

    ingest_time_out = f"Preparing documents took {ingest_toc - ingest_tic:0.1f} seconds"
    print(ingest_time_out)

    return doc_sections, "Finished splitting documents"

# # Functions for working with documents after loading them back in

def pull_out_data(series):

    # define a lambda function to convert each string into a tuple
    to_tuple = lambda x: eval(x)

    # apply the lambda function to each element of the series
    series_tup = series.apply(to_tuple)

    series_tup_content = list(zip(*series_tup))[1]

    series = pd.Series(list(series_tup_content))#.str.replace("^Main post content", "", regex=True).str.strip()

    return series

def docs_from_csv(df):

    import ast
    
    documents = []
    
    page_content = pull_out_data(df["0"])
    metadatas = pull_out_data(df["1"])

    for x in range(0,len(df)):       
        new_doc = Document(page_content=page_content[x], metadata=metadatas[x])
        documents.append(new_doc)
        
    return documents

def docs_from_lists(docs, metadatas):

    documents = []

    for x, doc in enumerate(docs):
        new_doc = Document(page_content=doc, metadata=metadatas[x])
        documents.append(new_doc)
        
    return documents

def docs_elements_from_csv_save(docs_path="documents.csv"):

    documents = pd.read_csv(docs_path)

    docs_out = docs_from_csv(documents)

    out_df = pd.DataFrame(docs_out)

    docs_content = pull_out_data(out_df[0].astype(str))

    docs_meta = pull_out_data(out_df[1].astype(str))

    doc_sources = [d['source'] for d in docs_meta]

    return out_df, docs_content, docs_meta, doc_sources

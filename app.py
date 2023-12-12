import nltk
from typing import TypeVar
nltk.download('names')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from search_funcs.fast_bm25 import BM25
from search_funcs.clean_funcs import initial_clean, get_lemma_tokens#, stem_sentence
from nltk import word_tokenize

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

import gradio as gr
import pandas as pd
import os

from itertools import compress

#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from transformers import AutoModel

import search_funcs.ingest as ing
import search_funcs.chatfuncs as chatf

# Import Chroma and instantiate a client. The default Chroma client is ephemeral, meaning it will not save to disk.
import chromadb
#from typing_extensions import Protocol
#from chromadb import Documents, EmbeddingFunction, Embeddings

# Remove Chroma database file. If it exists as it can cause issues
chromadb_file = "chroma.sqlite3"

if os.path.isfile(chromadb_file):
    os.remove(chromadb_file)

def prepare_input_data(in_file, text_column, clean="No", progress=gr.Progress()):

    filename = in_file.name
    # Import data

    df = read_file(filename)

    #df = pd.read_parquet(file_in.name)
    df_list = list(df[text_column].astype(str))
    #df_list = df

    if clean == "Yes":
        df_list_clean = initial_clean(df_list)

        # Save to file if you have cleaned the data
        out_file_name = save_prepared_data(in_file, df_list_clean, df, text_column)

        #corpus = [word_tokenize(doc.lower()) for doc in df_list_clean]
        corpus = [word_tokenize(doc.lower()) for doc in progress.tqdm(df_list_clean, desc = "Tokenising text", unit = "rows")]
        


    else: 
        #corpus = [word_tokenize(doc.lower()) for doc in df_list]
        corpus = [word_tokenize(doc.lower()) for doc in progress.tqdm(df_list, desc = "Tokenising text", unit = "rows")]
        out_file_name = None

    

    print("Finished data clean")

    if len(df_list) >= 20:
        message = "Data loaded"
    else:
        message = "Data loaded. Warning: dataset may be too short to get consistent search results."
    
    return corpus, message, df, out_file_name

def get_file_path_end(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    print(filename_without_extension)
    
    return filename_without_extension

def save_prepared_data(in_file, prepared_text_list, in_df, in_bm25_column):

    # Check if the list and the dataframe have the same length
    if len(prepared_text_list) != len(in_df):
        raise ValueError("The length of 'prepared_text_list' and 'in_df' must match.")
    
    file_end = ".parquet"

    file_name = get_file_path_end(in_file.name) + "_cleaned" + file_end

    prepared_text_df = pd.DataFrame(data={in_bm25_column + "_cleaned":prepared_text_list})

    # Drop original column from input file to reduce file size
    in_df = in_df.drop(in_bm25_column, axis = 1)

    prepared_df = pd.concat([in_df, prepared_text_df], axis = 1)

    if file_end == ".csv":
        prepared_df.to_csv(file_name)
    elif file_end == ".parquet":
        prepared_df.to_parquet(file_name)
    else: file_name = None
    

    return file_name

def prepare_bm25(corpus, k1=1.5, b = 0.75, alpha=-5):
    #bm25.save("saved_df_bm25")
    #bm25 = BM25.load(re.sub(r'\.pkl$', '', file_in.name))

    print("Preparing BM25 corpus")

    global bm25
    bm25 = BM25(corpus, k1=k1, b=b, alpha=alpha)

    message = "Search parameters loaded."

    print(message)

    return message

def convert_query_to_tokens(free_text_query, clean="No"):
    '''
    Split open text query into tokens and then lemmatise to get the core of the word
    '''  

    if clean=="Yes":
        split_query = word_tokenize(free_text_query.lower())
        out_query = get_lemma_tokens(split_query)
        #out_query = stem_sentence(free_text_query)
    else: 
        split_query = word_tokenize(free_text_query.lower())
        out_query = split_query

    return out_query

def bm25_search(free_text_query, in_no_search_results, original_data, text_column, clean = "No", in_join_file = None, in_join_column = "", search_df_join_column = ""):   

    # Prepare query
    if (clean == "Yes") | (text_column.endswith("_cleaned")):
        token_query = convert_query_to_tokens(free_text_query, clean="Yes")
    else:
        token_query = convert_query_to_tokens(free_text_query, clean="No")

    print(token_query)

    # Perform search
    print("Searching")

    results_index, results_text, results_scores = bm25.extract_documents_and_scores(token_query, bm25.corpus, n=in_no_search_results) #bm25.corpus #original_data[text_column]
    if not results_index:
        return "No search results found", None, token_query

    print("Search complete")

    # Prepare results and export
    joined_texts = [' '.join(inner_list) for inner_list in results_text]
    results_df = pd.DataFrame(data={"index": results_index,
                                    "search_text": joined_texts,
                                    "search_score_abs": results_scores})
    results_df['search_score_abs'] = abs(round(results_df['search_score_abs'], 2))
    results_df_out = results_df[['index', 'search_text', 'search_score_abs']].merge(original_data,left_on="index", right_index=True, how="left")#.drop("index", axis=1)
    
    # Join on additional files
    if in_join_file:
        join_filename = in_join_file.name

        # Import data
        join_df = read_file(join_filename)
        join_df[in_join_column] = join_df[in_join_column].astype(str).str.replace("\.0$","", regex=True)
        results_df_out[search_df_join_column] = results_df_out[search_df_join_column].astype(str).str.replace("\.0$","", regex=True)

        results_df_out = results_df_out.merge(join_df,left_on=search_df_join_column, right_on=in_join_column, how="left").drop(in_join_column, axis=1)
    

    # Reorder results by score
    results_df_out = results_df_out.sort_values('search_score_abs', ascending=False)



    # Out file
    results_df_name = "search_result.csv"
    results_df_out.to_csv(results_df_name, index= None)
    results_first_text = results_df_out[text_column].iloc[0]

    print("Returning results")

    return results_first_text, results_df_name, token_query

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'xlsx':
        return pd.read_excel(filename).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'parquet':
        return pd.read_parquet(filename).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")

def put_columns_in_df(in_file, in_bm25_column):
    '''
    When file is loaded, update the column dropdown choices and change 'clean data' dropdown option to 'no'.
    '''

    new_choices = []
    concat_choices = []
    
    
    df = read_file(in_file.name)
    new_choices = list(df.columns)

    print(new_choices)

    concat_choices.extend(new_choices)     
        
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(value="No", choices = ["Yes", "No"]),\
        gr.Dropdown(choices=concat_choices)

def put_columns_in_join_df(in_file, in_bm25_column):
    '''
    When file is loaded, update the column dropdown choices and change 'clean data' dropdown option to 'no'.
    '''

    print("in_bm25_column")

    new_choices = []
    concat_choices = []
    
    
    df = read_file(in_file.name)
    new_choices = list(df.columns)

    print(new_choices)

    concat_choices.extend(new_choices)     
        
    return gr.Dropdown(choices=concat_choices)

def dummy_function(gradio_component):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None    

def display_info(info_component):
    gr.Info(info_component)

# class MyEmbeddingFunction(EmbeddingFunction):
#     def __call__(self, input) -> Embeddings:
#         embeddings = []
#         for text in input:
#             embeddings.append(embeddings_model.encode(text))

#         return embeddings

def load_embeddings(embeddings_name = "jinaai/jina-embeddings-v2-small-en"):
    '''
    Load embeddings model and create a global variable based on it.
    '''

    # Import Chroma and instantiate a client. The default Chroma client is ephemeral, meaning it will not save to disk.
    
    #else: 
    embeddings_func = AutoModel.from_pretrained(embeddings_name, trust_remote_code=True)

    global embeddings

    embeddings = embeddings_func

    return embeddings

# Load embeddings
embeddings_name = "jinaai/jina-embeddings-v2-small-en"
#embeddings_name = "BAAI/bge-base-en-v1.5"
embeddings_model = AutoModel.from_pretrained(embeddings_name, trust_remote_code=True)
embeddings = load_embeddings(embeddings_name)

def docs_to_chroma_save(docs_out, embeddings = embeddings, progress=gr.Progress()):
    '''
    Takes a Langchain document class and saves it into a Chroma sqlite file.
    '''

    print(f"> Total split documents: {len(docs_out)}")

    #print(docs_out)

    page_contents = [doc.page_content for doc in docs_out]
    page_meta = [doc.metadata for doc in docs_out]
    ids_range = range(0,len(page_contents)) 
    ids = [str(element) for element in ids_range]

    embeddings_list = []
    for page in progress.tqdm(page_contents, desc = "Preparing search index", unit = "rows"):
        embeddings_list.append(embeddings.encode(sentences=page, max_length=1024).tolist())


    client = chromadb.PersistentClient(path=".")

    # Create a new Chroma collection to store the supporting evidence. We don't need to specify an embedding fuction, and the default will be used.
    try:
        collection = client.get_collection(name="my_collection")
        client.delete_collection(name="my_collection")
    except: 
        collection = client.create_collection(name="my_collection")
                                          
    collection.add(
    documents = page_contents,
    embeddings = embeddings_list,
    metadatas = page_meta,
    ids = ids)

    #chatf.vectorstore = vectorstore_func

    out_message = "Document processing complete"

    return out_message, collection

def jina_simple_retrieval(new_question_kworded, vectorstore, docs, k_val, out_passages,
                           vec_score_cut_off, vec_weight): # ,vectorstore, embeddings

            from numpy.linalg import norm

            cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

            query = embeddings.encode(new_question_kworded)

            # Calculate cosine similarity with each string in the list
            cosine_similarities = [cos_sim(query, string_vector) for string_vector in vectorstore]

            print(cosine_similarities)

            return cosine_similarities

def chroma_retrieval(new_question_kworded:str, vectorstore, docs, orig_df_col:str, k_val:int, out_passages:int,
                           vec_score_cut_off:float, vec_weight:float, in_join_file = None, in_join_column = None, search_df_join_column = None): # ,vectorstore, embeddings

            query = embeddings.encode(new_question_kworded).tolist()

            docs = vectorstore.query(
            query_embeddings=query,
            n_results= k_val # No practical limit on number of responses returned
            #where={"metadata_field": "is_equal_to_this"},
            #where_document={"$contains":"search_string"}
            )

            df_docs = pd.DataFrame(data={'ids': docs['ids'][0],
                                    'documents': docs['documents'][0],
                                    'metadatas':docs['metadatas'][0],
                                    'distances':docs['distances'][0]#,                                    
                                    #'embeddings': docs['embeddings']
                                    })
            
            def create_docs_keep_from_df(df):
                dict_out = {'ids' : [df['ids']],
                            'documents': [df['documents']],
                            'metadatas': [df['metadatas']],
                            'distances': [round(df['distances'].astype(float), 2)],
                            'embeddings': None
                            }
                return dict_out
                
            # Prepare the DataFrame by transposing
            #df_docs = df#.apply(lambda x: x.explode()).reset_index(drop=True)

            # Keep only documents with a certain score
            
            docs_scores = df_docs["distances"] #.astype(float)

            # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
            score_more_limit = df_docs.loc[docs_scores < vec_score_cut_off, :]
            docs_keep = create_docs_keep_from_df(score_more_limit) #list(compress(docs, score_more_limit))

            #print(docs_keep)

            if not docs_keep:
                return 'No result found!', ""

            # Only keep sources that are at least 100 characters long
            docs_len = score_more_limit["documents"].str.len() >= 100
            length_more_limit = score_more_limit.loc[docs_len, :] #pd.Series(docs_len) >= 100
            docs_keep = create_docs_keep_from_df(length_more_limit) #list(compress(docs_keep, length_more_limit))

            #print(length_more_limit)

            if not docs_keep:
                return 'No result found!', ""
            
            length_more_limit['ids'] = length_more_limit['ids'].astype(int)

            #length_more_limit.to_csv("length_more_limit.csv", index = None)

            # Explode the 'metadatas' dictionary into separate columns
            df_metadata_expanded = df_docs['metadatas'].apply(pd.Series)

            # Concatenate the original DataFrame with the expanded metadata DataFrame
            results_df_out = pd.concat([df_docs.drop('metadatas', axis=1), df_metadata_expanded], axis=1)

            results_df_out = results_df_out.rename(columns={"documents":orig_df_col})

            results_df_out = results_df_out.drop(["page_section", "row", "source", "id"], axis=1, errors="ignore")
            results_df_out['distances'] = round(results_df_out['distances'].astype(float), 2)

            # Join back to original df
            # results_df_out = orig_df.merge(length_more_limit[['ids', 'distances']], left_index = True, right_on = "ids", how="inner").sort_values("distances")

            # Join on additional files
            if in_join_file:
                join_filename = in_join_file.name

                # Import data
                join_df = read_file(join_filename)
                join_df[in_join_column] = join_df[in_join_column].astype(str).str.replace("\.0$","", regex=True)
                results_df_out[search_df_join_column] = results_df_out[search_df_join_column].astype(str).str.replace("\.0$","", regex=True)

                results_df_out = results_df_out.merge(join_df,left_on=search_df_join_column, right_on=in_join_column, how="left").drop(in_join_column, axis=1)


            results_df_name = "semantic_search_result.csv"
            results_df_out.to_csv(results_df_name, index= None)
            results_first_text = results_df_out[orig_df_col][0]

            return results_first_text, results_df_name

## Gradio app - BM25 search
block = gr.Blocks(theme = gr.themes.Base())

with block:

    ingest_text = gr.State()
    ingest_metadata = gr.State()
    ingest_docs = gr.State()
    vectorstore_state = gr.State() # globals()["vectorstore"]
    embeddings_state = gr.State() # globals()["embeddings"]

    k_val = gr.State(9999)
    out_passages = gr.State(9999)
    vec_score_cut_off = gr.State(100)
    vec_weight = gr.State(1)

    docs_keep_as_doc_state = gr.State()
    doc_df_state = gr.State()
    docs_keep_out_state = gr.State()

    corpus_state = gr.State()
    data_state = gr.State(pd.DataFrame())

    in_k1_info = gr.State("""k1: Constant used for influencing the term frequency saturation. After saturation is reached, additional
presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
the type of documents or queries. Information taken from https://github.com/Inspirateur/Fast-BM25""")
    in_b_info = gr.State("""b: Constant used for influencing the effects of different document lengths relative to average document length.
When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
[1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
depends on factors such as the type of documents or queries. Information taken from https://github.com/Inspirateur/Fast-BM25""")
    in_alpha_info = gr.State("""alpha: IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy of BM25 but increase performance. Information taken from https://github.com/Inspirateur/Fast-BM25""")
    in_no_search_info = gr.State("""Search results number: Maximum number of search results that will be returned. Bear in mind that if the alpha value is greater than the minimum, common words will be removed from the dataset, and so the number of search results returned may be lower than this value.""")
    in_clean_info = gr.State("""Clean text: Clean the input text and search query. The function will try to remove email components and tags, and then will 'stem' the words. I.e. it will remove the endings of words (e.g. smashed becomes smash) so that the search engine is looking for the common 'core' of words between the query and dataset.""")

    gr.Markdown(
    """
    # Fast text search
    Enter a text query below to search through a text data column and find relevant terms. It will only find terms containing the exact text you enter. Your data should contain at least 20 entries for the search to consistently return results.
    """)
    
    with gr.Tab(label="Keyword search"):
        with gr.Row():
            current_source = gr.Textbox(label="Current data source(s)", value="None")

        with gr.Accordion(label = "Load in data", open=True):
            in_bm25_file = gr.File(label="Upload your search data here")
            with gr.Row():
                in_bm25_column = gr.Dropdown(label="Enter the name of the text column in the data file to search")
                
                load_bm25_data_button = gr.Button(value="Load data")
                 
            with gr.Row():
                load_finished_message = gr.Textbox(label="Load progress", scale = 2)

        with gr.Accordion(label = "Search data", open=True):
            with gr.Row():
                in_query = gr.Textbox(label="Enter your search term")
                mod_query = gr.Textbox(label="Cleaned search term (the terms that are passed to the search engine)")
                             
            search_button = gr.Button(value="Search text")

            with gr.Row():
                output_single_text = gr.Textbox(label="Top result")
                output_file = gr.File(label="File output")

    with gr.Tab("Fuzzy/semantic search"):
        with gr.Row():
            current_source_semantic = gr.Textbox(label="Current data source(s)", value="None")

        with gr.Accordion("Load in data", open = True):
            in_semantic_file = gr.File(label="Upload data file for semantic search")
            in_semantic_column = gr.Dropdown(label="Enter the name of the text column in the data file to search")
            load_semantic_data_button = gr.Button(value="Load in CSV/Excel file", variant="secondary", scale=0)
        
        ingest_embed_out = gr.Textbox(label="File/web page preparation progress")
        semantic_query = gr.Textbox(label="Enter semantic search query here")
        semantic_submit = gr.Button(value="Start semantic search", variant="secondary", scale = 1)

        with gr.Row():
            semantic_output_single_text = gr.Textbox(label="Top result")
            semantic_output_file = gr.File(label="File output")
            
    with gr.Tab(label="Advanced options"):
        with gr.Accordion(label="Data load / save options", open = False):
            #with gr.Row():
            in_clean_data = gr.Dropdown(label = "Clean text during load (remove tags, stem words). This will take some time!", value="No", choices=["Yes", "No"])
            #save_clean_data_button = gr.Button(value = "Save loaded data to file", scale = 1)
        with gr.Accordion(label="Search options", open = False):
            with gr.Row():
                in_k1 = gr.Slider(label = "k1 value", value = 1.5, minimum = 0.1, maximum = 5, step = 0.1, scale = 3)
                in_k1_button = gr.Button(value = "k1 value info", scale = 1)
            with gr.Row():
                in_b = gr.Slider(label = "b value", value = 0.75, minimum = 0.1, maximum = 5, step = 0.05, scale = 3)
                in_b_button = gr.Button(value = "b value info", scale = 1)
            with gr.Row():
                in_alpha = gr.Slider(label = "alpha value / IDF cutoff", value = -5, minimum = -5, maximum = 10, step = 1, scale = 3)
                in_alpha_button = gr.Button(value = "alpha value info", scale = 1)
            with gr.Row():
                in_no_search_results = gr.Slider(label="Maximum number of search results to return", value = 100000, minimum=10, maximum=100000, step=10, scale = 3)
                in_no_search_results_button = gr.Button(value = "Search results number info", scale = 1)
            with gr.Row():
                in_search_param_button = gr.Button(value="Load search parameters (Need to click this if you changed anything above)")
        with gr.Accordion(label = "Join on additional dataframes to results", open = False):
            in_join_file = gr.File(label="Upload your data to join here")
            in_join_column = gr.Dropdown(label="Column to join in new data frame")
            search_df_join_column = gr.Dropdown(label="Column to join in search data frame")

        in_search_param_button.click(fn=prepare_bm25, inputs=[corpus_state, in_k1, in_b, in_alpha], outputs=[load_finished_message])
                      
    # ---
    in_k1_button.click(display_info, inputs=in_k1_info)
    in_b_button.click(display_info, inputs=in_b_info)
    in_alpha_button.click(display_info, inputs=in_alpha_info)
    in_no_search_results_button.click(display_info, inputs=in_no_search_info)
    

    # Update dropdowns upon initial file load
    in_bm25_file.upload(put_columns_in_df, inputs=[in_bm25_file, in_bm25_column], outputs=[in_bm25_column, in_clean_data, search_df_join_column])
    in_join_file.upload(put_columns_in_join_df, inputs=[in_join_file, in_join_column], outputs=[in_join_column])
 
    # Load in BM25 data
    load_bm25_data_button.click(fn=prepare_input_data, inputs=[in_bm25_file, in_bm25_column, in_clean_data], outputs=[corpus_state, load_finished_message, data_state, output_file]).\
    then(fn=prepare_bm25, inputs=[corpus_state, in_k1, in_b, in_alpha], outputs=[load_finished_message]).\
    then(fn=put_columns_in_df, inputs=[in_bm25_file, in_bm25_column], outputs=[in_bm25_column, in_clean_data, search_df_join_column])
   
    # BM25 search functions on click or enter
    search_button.click(fn=bm25_search, inputs=[in_query, in_no_search_results, data_state, in_bm25_column, in_clean_data, in_join_file, in_join_column, search_df_join_column], outputs=[output_single_text, output_file, mod_query], api_name="search")
    in_query.submit(fn=bm25_search, inputs=[in_query, in_no_search_results, data_state, in_bm25_column, in_clean_data, in_join_file, in_join_column, search_df_join_column], outputs=[output_single_text, output_file, mod_query])
    
    # Load in a csv/excel file for semantic search
    in_semantic_file.upload(put_columns_in_df, inputs=[in_semantic_file, in_semantic_column], outputs=[in_semantic_column, in_clean_data, search_df_join_column])
    load_semantic_data_button.click(ing.parse_csv_or_excel, inputs=[in_semantic_file, in_semantic_column], outputs=[ingest_text, current_source_semantic]).\
             then(ing.csv_excel_text_to_docs, inputs=[ingest_text, in_semantic_column], outputs=[ingest_docs]).\
             then(docs_to_chroma_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state])
    
    # Semantic search query
    semantic_submit.click(chroma_retrieval, inputs=[semantic_query, vectorstore_state, ingest_docs, in_semantic_column, k_val, out_passages, vec_score_cut_off, vec_weight, in_join_file, in_join_column, search_df_join_column], outputs=[semantic_output_single_text, semantic_output_file], api_name="semantic")
    
    # Dummy functions just to get dropdowns to work correctly with Gradio 3.50
    in_bm25_column.change(dummy_function, in_bm25_column, None)
    search_df_join_column.change(dummy_function, search_df_join_column, None)
    in_join_column.change(dummy_function, in_join_column, None)
    in_semantic_column.change(dummy_function, in_join_column, None)

block.queue().launch(debug=True)



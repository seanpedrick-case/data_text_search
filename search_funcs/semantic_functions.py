import time
import pandas as pd
from typing import Type
import gradio as gr
import numpy as np
from datetime import datetime
from search_funcs.helper_functions import get_file_path_end, create_highlighted_excel_wb, ensure_output_folder_exists, output_folder
PandasDataFrame = Type[pd.DataFrame]
today_rev = datetime.now().strftime("%Y%m%d")

def load_embedding_model(embeddings_name = "BAAI/bge-small-en-v1.5", embedding_loc="bge/"):

    from torch import cuda, backends
    from sentence_transformers import SentenceTransformer

    # Check for torch cuda
    print("Is CUDA enabled? ", cuda.is_available())
    print("Is a CUDA device available on this computer?", backends.cudnn.enabled)
    if cuda.is_available():
        torch_device = "cuda"
        #os.system("nvidia-smi")
    else: 
        torch_device =  "cpu"

    print("Device used is: ", torch_device)

    # Define a list of possible local locations to search for the model
    local_embeddings_locations = [
        "model/" + embedding_loc, # Potential local location
        "/model/" + embedding_loc, # Potential location in Docker container
        "/home/user/app/model/" + embedding_loc # This is inside a Docker container
    ]

    # Attempt to load the model from each local location
    for location in local_embeddings_locations:
        try:
            embeddings_model = SentenceTransformer(location)
            print(f"Found local model installation at: {location}")
            break  # Exit the loop if the model is found
        except Exception as e:
            print(f"Failed to load model from {location}: {e}")
            continue
    else:
        # If the loop completes without finding the model in any local location
        embeddings_model = SentenceTransformer(embeddings_name)
        print("Could not find local model installation. Downloading from Huggingface")

    # Load the sentence transformer model and move it to CPU/GPU
    embeddings_model = embeddings_model.to(torch_device)

    return embeddings_model, torch_device

def docs_to_bge_embed_np_array(
    docs_out: list, 
    in_file: list, 
    output_file_state: str, 
    clean: str,
    embeddings_state: np.ndarray,
    embeddings_model_name:str,
    embeddings_model_loc:str,
    return_intermediate_files: str = "No", 
    embeddings_compress: str = "No",    
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> tuple:
    """
    Process documents to create BGE embeddings and save them as a numpy array.

    Parameters:
    - docs_out (list): List of documents to be embedded.
    - in_file (list): List of input files.
    - output_file_state (str): State of the output file.
    - clean (str): Indicates if the data should be cleaned.
    - embeddings_state (np.ndarray): Current state of embeddings.
    - embeddings_model_name (str): The Huggingface repo name of the embeddings model.
    - embeddings_model_loc (str): Embeddings model save location.
    - return_intermediate_files (str, optional): Whether to return intermediate files. Default is "No".
    - embeddings_compress (str, optional): Whether to compress the embeddings to int8 precision. Default is "No".    
    - progress (gr.Progress, optional): Progress tracker for the function. Default is gr.Progress(track_tqdm=True).

    Returns:
    - tuple: A tuple containing the output message, embeddings, and output file state.
    """

    embeddings_model, torch_device = load_embedding_model(embeddings_model_name, embeddings_model_loc)

    ensure_output_folder_exists(output_folder)

    if not in_file:
        out_message = "No input file found. Please load in at least one file."
        print(out_message)
        return out_message, None, None, output_file_state        

    progress(0.6, desc = "Loading/creating embeddings")

    print(f"> Total split documents: {len(docs_out)}")

    page_contents = [doc.page_content for doc in docs_out]

    ## Load in pre-embedded file if exists
    file_list = [string.name for string in in_file]

    embeddings_file_names = [string for string in file_list if "embedding" in string.lower()]
    data_file_names = [string for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower()]# and "gz" not in string.lower()]
    data_file_name = data_file_names[0]
    data_file_name_no_ext = get_file_path_end(data_file_name)

    out_message = "Document processing complete. Ready to search."

    if embeddings_state.size == 0:
        tic = time.perf_counter()
        print("Starting to embed documents.")

        # Encode embeddings. If in normal mode, float32, if in 'super compress' mode, int8
        batch_size = 32

        if "bge" in embeddings_model_name:
            print("Embedding with BGE model")
        else:
            print("Embedding with MiniLM-L6-v2 model")

        if embeddings_compress == "No":
            print("Embedding with full fp32 precision")
            embeddings_out = embeddings_model.encode(sentences=page_contents, show_progress_bar = True, batch_size = batch_size)
        else:
            print("Embedding with int8 precision")
            embeddings_out = embeddings_model.encode(sentences=page_contents, show_progress_bar = True, batch_size = batch_size, precision="int8")

        toc = time.perf_counter()
        time_out = f"The embedding took {toc - tic:0.1f} seconds"
        print(time_out)

        # If you want to save your files for next time
        if return_intermediate_files == "Yes":
            if clean == "Yes": data_file_name_no_ext = data_file_name_no_ext + "_cleaned"
            else: data_file_name_no_ext = data_file_name_no_ext

            progress(0.9, desc = "Saving embeddings to file")
            if embeddings_compress == "No":
                semantic_search_file_name = output_folder + data_file_name_no_ext + '_bge_embeddings.npz'
            else:
                semantic_search_file_name = output_folder + data_file_name_no_ext + '_bge_embedding_compress.npz'
                
            np.savez_compressed(semantic_search_file_name, embeddings_out)

            output_file_state.append(semantic_search_file_name)

            return out_message, embeddings_out, output_file_state, output_file_state, embeddings_model

        return out_message, embeddings_out, output_file_state, output_file_state, embeddings_model
    else:
        # Just return existing embeddings if already exist
        embeddings_out = embeddings_state
    
    print(out_message)

    return out_message, embeddings_out, output_file_state, output_file_state, embeddings_model

def process_data_from_scores_df(
    df_docs: pd.DataFrame, 
    in_join_file: pd.DataFrame, 
    vec_score_cut_off: float, 
    in_join_column: str, 
    search_df_join_column: str, 
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> pd.DataFrame:
    """
    Process the data from the scores DataFrame by filtering based on score cutoff and document length,
    and optionally joining with an additional file.

    Parameters
    ----------
    df_docs : pd.DataFrame
        DataFrame containing document scores and metadata.
    in_join_file : pd.DataFrame
        DataFrame to join with the results based on specified columns.
    vec_score_cut_off : float
        Cutoff value for the vector similarity score.
    in_join_column : str
        Column name in the join file to join on.
    search_df_join_column : str
        Column name in the search DataFrame to join on.
    progress : gr.Progress, optional
        Progress tracker for the function (default is gr.Progress(track_tqdm=True)).

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with filtered and joined data.
    """
           
    docs_scores = df_docs["distances"] #.astype(float)

    # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
    score_more_limit = df_docs.loc[docs_scores > vec_score_cut_off, :]

    if score_more_limit.empty:
        return pd.DataFrame()

    # Only keep sources that are at least 100 characters long
    docs_len = score_more_limit["documents"].str.len() >= 100

    length_more_limit = score_more_limit.loc[docs_len == True, :] #pd.Series(docs_len) >= 100

    if length_more_limit.empty:
        return pd.DataFrame()
            
    length_more_limit['ids'] = length_more_limit['ids'].astype(int)


    # Explode the 'metadatas' dictionary into separate columns
    df_metadata_expanded = length_more_limit['metadatas'].apply(pd.Series)

    # Concatenate the original DataFrame with the expanded metadata DataFrame
    results_df_out = pd.concat([length_more_limit.drop('metadatas', axis=1), df_metadata_expanded], axis=1)

    results_df_out = results_df_out.rename(columns={"documents":"search_text"})

    results_df_out = results_df_out.drop(["page_section", "row", "source", "id"], axis=1, errors="ignore")
    results_df_out['distances'] = round(results_df_out['distances'].astype(float), 3)
    

    # Join on additional files
    if not in_join_file.empty:
        progress(0.5, desc = "Joining on additional data file")
        join_df = in_join_file

        join_df[in_join_column] = join_df[in_join_column].astype(str).str.replace("\.0$","", regex=True)

        # Duplicates dropped so as not to expand out dataframe
        join_df = join_df.drop_duplicates(in_join_column)

        results_df_out[search_df_join_column] = results_df_out[search_df_join_column].astype(str).str.replace("\.0$","", regex=True)

        results_df_out = results_df_out.merge(join_df,left_on=search_df_join_column, right_on=in_join_column, how="left", suffixes=('','_y'))#.drop(in_join_column, axis=1)

    return results_df_out

def bge_semantic_search(
    query_str: str, 
    embeddings: np.ndarray, 
    documents: list, 
    k_val: int, 
    vec_score_cut_off: float,
    embeddings_model,
    embeddings_model_name: str,
    embeddings_compress:str,
    in_join_file: pd.DataFrame, 
    in_join_column: str = None, 
    search_df_join_column: str = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> tuple:
    """
    Perform a semantic search using the BGE model.

    Parameters:
    - query_str (str): The query string to search for.
    - embeddings (np.ndarray): The embeddings to search within.
    - documents (list): The list of documents to search.
    - k_val (int): The number of top results to return.
    - vec_score_cut_off (float): The score cutoff for filtering results.
    - embeddings_model (SentenceTransformer, optional): The embeddings model to use.
    - embeddings_model_name (str): The Huggingface repo name of the embeddings model.
    - embeddings_compress (str): Whether the embeddings have been compressed to int8 precision
    - in_join_file (pd.DataFrame): The DataFrame to join with the search results.
    - in_join_column (str, optional): The column name in the join DataFrame to join on. Default is None.
    - search_df_join_column (str, optional): The column name in the search DataFrame to join on. Default is None.  
    - progress (gr.Progress, optional): Progress tracker for the function. Default is gr.Progress(track_tqdm=True).

    Returns:
    - tuple: The DataFrame containing the search results.
    """

    progress(0, desc = "Conducting semantic search")

    output_files = []

    ensure_output_folder_exists(output_folder)

    print("Searching")

    from sentence_transformers import quantize_embeddings

    # Encode the query using the sentence transformer and convert to a PyTorch tensor
    if "bge" in embeddings_model_name:
        print("Comparing similarity using BGE model")
    else:
        print("Comparing similarity using MiniLM-L6-v2 model")


    if embeddings_compress == "Yes":
        query_fp32 = embeddings_model.encode(query_str)

        # Using a query as int8 doesn't actually seem to work
        # query_int8 = quantize_embeddings(
        #     query_fp32, precision="int8", calibration_embeddings=embeddings
        # )
        
    else:
        query_fp32 = embeddings_model.encode(query_str)
  
    #print("query:", query_fp32)
    #print("embeddings:", embeddings)

    # Normalise embeddings

    query = query_fp32.astype('float32')

    query_norm = np.linalg.norm(query)
    normalized_query = query / query_norm

    embeddings = embeddings.astype('float32')

    embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)  # Keep dims to allow broadcasting
    normalized_embeddings = embeddings / embeddings_norm

    #print("normalized_query:", normalized_query)
    #print("normalized_embeddings:", normalized_embeddings)

    cosine_similarities = (normalized_query @ normalized_embeddings.T)

    #print("Initial cosine similarities:", cosine_similarities)

    # Create a Pandas Series
    cosine_similarities_series = pd.Series(cosine_similarities)

    # Pull out relevent info from documents
    page_contents = [doc.page_content for doc in documents]
    page_meta = [doc.metadata for doc in documents]
    ids_range = range(0,len(page_contents)) 
    ids = [str(element) for element in ids_range]

    df_documents = pd.DataFrame(data={"ids": ids,
                                "documents": page_contents,
                                    "metadatas":page_meta,
                                    "distances":cosine_similarities_series}).sort_values("distances", ascending=False).iloc[0:k_val,:]

    
    results_df_out = process_data_from_scores_df(df_documents, in_join_file, vec_score_cut_off, in_join_column, search_df_join_column)

    print("Search complete")

    # If nothing found, return error message
    if results_df_out.empty:
        return 'No result found!', None
    
    query_str_file = query_str.replace(" ", "_")

    results_df_name = output_folder + "semantic_search_result_" + today_rev + "_" +  query_str_file + ".xlsx"

    print("Saving search output to file")
    progress(0.7, desc = "Saving search output to file")

    # Highlight found text and save to file
    results_df_out_wb = create_highlighted_excel_wb(results_df_out, query_str, "search_text")
    results_df_out_wb.save(results_df_name)

    #results_df_out.to_excel(results_df_name, index= None)
    results_first_text = results_df_out.iloc[0, 1]
    output_files.append(results_df_name)

    #csv_output_file = output_folder + "semantic_search_result_" + today_rev + "_" +  query_str_file + ".csv" 
    #results_df_out.to_csv(csv_output_file, index=None)
    #output_files.append(csv_output_file)

    print("Returning results")

    return results_first_text, output_files
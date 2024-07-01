import os
import time
import pandas as pd
from typing import Type
import gradio as gr
import numpy as np
from datetime import datetime
#from transformers import AutoModel, AutoTokenizer
from search_funcs.helper_functions import get_file_path_end
#import torch
from torch import cuda, backends#, tensor, mm, utils
from sentence_transformers import SentenceTransformer

today_rev = datetime.now().strftime("%Y%m%d")

# Check for torch cuda
print("Is CUDA enabled? ", cuda.is_available())
print("Is a CUDA device available on this computer?", backends.cudnn.enabled)
if cuda.is_available():
    torch_device = "cuda"
    os.system("nvidia-smi")

else: 
    torch_device =  "cpu"

print("Device used is: ", torch_device)

from search_funcs.helper_functions import create_highlighted_excel_wb, ensure_output_folder_exists, output_folder

PandasDataFrame = Type[pd.DataFrame]

# Load embeddings - Jina - deprecated
# Pinning a Jina revision for security purposes: https://www.baseten.co/blog/pinning-ml-model-revisions-for-compatibility-and-security/
# Save Jina model locally as described here: https://huggingface.co/jinaai/jina-embeddings-v2-base-en/discussions/29
# embeddings_name = "jinaai/jina-embeddings-v2-small-en"
# local_embeddings_location = "model/jina/"
# revision_choice = "b811f03af3d4d7ea72a7c25c802b21fc675a5d99"

# try:
#     embeddings_model = AutoModel.from_pretrained(local_embeddings_location, revision = revision_choice, trust_remote_code=True,local_files_only=True, device_map="auto")
# except:
#     embeddings_model = AutoModel.from_pretrained(embeddings_name, revision = revision_choice, trust_remote_code=True, device_map="auto")

# Load embeddings
embeddings_name = "BAAI/bge-small-en-v1.5"

# Define a list of possible local locations to search for the model
local_embeddings_locations = [
    "model/bge/", # Potential local location
    "/model/bge/", # Potential location in Docker container
    "/home/user/app/model/bge/" # This is inside a Docker container
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
    
def docs_to_bge_embed_np_array(docs_out, in_file, embeddings_state, output_file_state, clean, return_intermediate_files = "No", embeddings_super_compress = "No", embeddings_model = embeddings_model, progress=gr.Progress(track_tqdm=True)):
    '''
    Takes a Langchain document class and saves it into a Numpy array.
    '''

    ensure_output_folder_exists(output_folder)

    if not in_file:
        out_message = "No input file found. Please load in at least one file."
        print(out_message)
        return out_message, None, None, output_file_state
        

    progress(0.6, desc = "Loading/creating embeddings")

    print(f"> Total split documents: {len(docs_out)}")

    #print(docs_out)

    page_contents = [doc.page_content for doc in docs_out]

    ## Load in pre-embedded file if exists
    file_list = [string.name for string in in_file]

    #print(file_list)

    embeddings_file_names = [string for string in file_list if "embedding" in string.lower()]
    data_file_names = [string for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower()]# and "gz" not in string.lower()]
    data_file_name = data_file_names[0]
    data_file_name_no_ext = get_file_path_end(data_file_name)

    out_message = "Document processing complete. Ready to search."

     # print("embeddings loaded: ", embeddings_out)

    if embeddings_state.size == 0:
        tic = time.perf_counter()
        print("Starting to embed documents.")
        #embeddings_list = []
        #for page in progress.tqdm(page_contents, desc = "Preparing search index", unit = "rows"):
        #    embeddings_list.append(embeddings.encode(sentences=page, max_length=1024).tolist())

        
        
        #embeddings_out = calc_bge_norm_embeddings(page_contents, embeddings_model, tokenizer)

        embeddings_out = embeddings_model.encode(sentences=page_contents, show_progress_bar = True, batch_size = 32, normalize_embeddings=True) # For BGE
        #embeddings_list = embeddings.encode(sentences=page_contents, normalize_embeddings=True).tolist() # For BGE embeddings
        #embeddings_list = embeddings.encode(sentences=page_contents).tolist() # For minilm

        toc = time.perf_counter()
        time_out = f"The embedding took {toc - tic:0.1f} seconds"
        print(time_out)

        # If you want to save your files for next time
        if return_intermediate_files == "Yes":
            if clean == "Yes": data_file_name_no_ext = data_file_name_no_ext + "_cleaned"
            else: data_file_name_no_ext = data_file_name_no_ext

            progress(0.9, desc = "Saving embeddings to file")
            if embeddings_super_compress == "No":
                semantic_search_file_name = output_folder + data_file_name_no_ext + '_bge_embeddings.npz'
                np.savez_compressed(semantic_search_file_name, embeddings_out)
            else:
                semantic_search_file_name = output_folder + data_file_name_no_ext + '_bge_embedding_compress.npz'
                embeddings_out_round = np.round(embeddings_out, 3) 
                embeddings_out_round *= 100 # Rounding not currently used
                np.savez_compressed(semantic_search_file_name, embeddings_out_round)

            output_file_state.append(semantic_search_file_name)

            return out_message, embeddings_out, output_file_state, output_file_state

        return out_message, embeddings_out, output_file_state, output_file_state
    else:
        # Just return existing embeddings if already exist
        embeddings_out = embeddings_state
    
    print(out_message)

    return out_message, embeddings_out, output_file_state, output_file_state

def process_data_from_scores_df(df_docs, in_join_file, out_passages, vec_score_cut_off, vec_weight, orig_df_col, in_join_column, search_df_join_column, progress = gr.Progress(track_tqdm=True)):

    def create_docs_keep_from_df(df):
        dict_out = {'ids' : [df['ids']],
                    'documents': [df['documents']],
                    'metadatas': [df['metadatas']],
                    'distances': [round(df['distances'].astype(float), 4)],
                    'embeddings': None
                    }
        return dict_out
        
    # Prepare the DataFrame by transposing
    #df_docs = df#.apply(lambda x: x.explode()).reset_index(drop=True)

    # Keep only documents with a certain score

    #print(df_docs)
    
    docs_scores = df_docs["distances"] #.astype(float)

    # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
    score_more_limit = df_docs.loc[docs_scores > vec_score_cut_off, :]
    #docs_keep = create_docs_keep_from_df(score_more_limit) #list(compress(docs, score_more_limit))

    #print(docs_keep)

    if score_more_limit.empty:
        return pd.DataFrame()

    # Only keep sources that are at least 100 characters long
    docs_len = score_more_limit["documents"].str.len() >= 100

    #print(docs_len)

    length_more_limit = score_more_limit.loc[docs_len == True, :] #pd.Series(docs_len) >= 100
    #docs_keep = create_docs_keep_from_df(length_more_limit) #list(compress(docs_keep, length_more_limit))

    #print(length_more_limit)

    if length_more_limit.empty:
        return pd.DataFrame()
            
    length_more_limit['ids'] = length_more_limit['ids'].astype(int)

    #length_more_limit.to_csv("length_more_limit.csv", index = None)

    # Explode the 'metadatas' dictionary into separate columns
    df_metadata_expanded = length_more_limit['metadatas'].apply(pd.Series)

    #print(length_more_limit)
    #print(df_metadata_expanded)

    # Concatenate the original DataFrame with the expanded metadata DataFrame
    results_df_out = pd.concat([length_more_limit.drop('metadatas', axis=1), df_metadata_expanded], axis=1)

    results_df_out = results_df_out.rename(columns={"documents":"search_text"})

    results_df_out = results_df_out.drop(["page_section", "row", "source", "id"], axis=1, errors="ignore")
    results_df_out['distances'] = round(results_df_out['distances'].astype(float), 3)
    

    # Join back to original df
    # results_df_out = orig_df.merge(length_more_limit[['ids', 'distances']], left_index = True, right_on = "ids", how="inner").sort_values("distances")

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

def bge_simple_retrieval(query_str:str, vectorstore, docs, orig_df_col:str, k_val:int, out_passages:int,
                           vec_score_cut_off:float, vec_weight:float, in_join_file, in_join_column = None, search_df_join_column = None, device = torch_device, embeddings = embeddings_model, progress=gr.Progress(track_tqdm=True)): # ,vectorstore, embeddings

    # print("vectorstore loaded: ", vectorstore)
    progress(0, desc = "Conducting semantic search")

    ensure_output_folder_exists(output_folder)

    print("Searching")

    # Convert it to a PyTorch tensor and transfer to GPU
    #vectorstore_tensor = tensor(vectorstore).to(device)

    # Load the sentence transformer model and move it to GPU
    embeddings = embeddings.to(device)

    # Encode the query using the sentence transformer and convert to a PyTorch tensor
    query = embeddings.encode(query_str, normalize_embeddings=True)

    # query = calc_bge_norm_embeddings(query_str, embeddings_model=embeddings_model, tokenizer=tokenizer)

    #query_tensor = tensor(query).to(device)

    # if query_tensor.dim() == 1:
    #     query_tensor = query_tensor.unsqueeze(0)  # Reshape to 2D with one row

    # Sentence transformers method, not used:
    cosine_similarities = query @ vectorstore.T
    #cosine_similarities = util.cos_sim(query_tensor, vectorstore_tensor)[0]
    #top_results = torch.topk(cos_scores, k=top_k)
   

    # Normalize the query tensor and vectorstore tensor
    #query_norm = query_tensor / query_tensor.norm(dim=1, keepdim=True)
    #vectorstore_norm = vectorstore_tensor / vectorstore_tensor.norm(dim=1, keepdim=True)

    # Calculate cosine similarities (batch processing)
    #cosine_similarities = mm(query_norm, vectorstore_norm.T)
    #cosine_similarities = mm(query_tensor, vectorstore_tensor.T)

    # Flatten the tensor to a 1D array
    cosine_similarities = cosine_similarities.flatten()

    # Convert to a NumPy array if it's still a PyTorch tensor
    #cosine_similarities = cosine_similarities.cpu().numpy()

    # Create a Pandas Series
    cosine_similarities_series = pd.Series(cosine_similarities)

    # Pull out relevent info from docs
    page_contents = [doc.page_content for doc in docs]
    page_meta = [doc.metadata for doc in docs]
    ids_range = range(0,len(page_contents)) 
    ids = [str(element) for element in ids_range]

    df_docs = pd.DataFrame(data={"ids": ids,
                                "documents": page_contents,
                                    "metadatas":page_meta,
                                    "distances":cosine_similarities_series}).sort_values("distances", ascending=False).iloc[0:k_val,:]

    
    results_df_out = process_data_from_scores_df(df_docs, in_join_file, out_passages, vec_score_cut_off, vec_weight, orig_df_col, in_join_column, search_df_join_column)

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

    print("Returning results")

    return results_first_text, results_df_name


def docs_to_jina_embed_np_array_deprecated(docs_out, in_file, embeddings_state, return_intermediate_files = "No", embeddings_super_compress = "No", embeddings = embeddings_model, progress=gr.Progress(track_tqdm=True)):
    '''
    Takes a Langchain document class and saves it into a Chroma sqlite file.
    '''
    if not in_file:
        out_message = "No input file found. Please load in at least one file."
        print(out_message)
        return out_message, None, None
        

    progress(0.6, desc = "Loading/creating embeddings")

    print(f"> Total split documents: {len(docs_out)}")

    #print(docs_out)

    page_contents = [doc.page_content for doc in docs_out]

    ## Load in pre-embedded file if exists
    file_list = [string.name for string in in_file]

    #print(file_list)

    embeddings_file_names = [string for string in file_list if "embedding" in string.lower()]
    data_file_names = [string for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower()]# and "gz" not in string.lower()]
    data_file_name = data_file_names[0]
    data_file_name_no_ext = get_file_path_end(data_file_name)

    out_message = "Document processing complete. Ready to search."

     # print("embeddings loaded: ", embeddings_out)

    if embeddings_state.size == 0:
        tic = time.perf_counter()
        print("Starting to embed documents.")
        #embeddings_list = []
        #for page in progress.tqdm(page_contents, desc = "Preparing search index", unit = "rows"):
        #    embeddings_list.append(embeddings.encode(sentences=page, max_length=1024).tolist())

        embeddings_out = embeddings.encode(sentences=page_contents, max_length=1024, show_progress_bar = True, batch_size = 32) # For Jina embeddings
        #embeddings_list = embeddings.encode(sentences=page_contents, normalize_embeddings=True).tolist() # For BGE embeddings
        #embeddings_list = embeddings.encode(sentences=page_contents).tolist() # For minilm

        toc = time.perf_counter()
        time_out = f"The embedding took {toc - tic:0.1f} seconds"
        print(time_out)

        # If you want to save your files for next time
        if return_intermediate_files == "Yes":
            progress(0.9, desc = "Saving embeddings to file")
            if embeddings_super_compress == "No":
                semantic_search_file_name = data_file_name_no_ext + '_' + 'embeddings.npz'
                np.savez_compressed(semantic_search_file_name, embeddings_out)
            else:
                semantic_search_file_name = data_file_name_no_ext + '_' + 'embedding_compress.npz'
                embeddings_out_round = np.round(embeddings_out, 3) 
                embeddings_out_round *= 100 # Rounding not currently used
                np.savez_compressed(semantic_search_file_name, embeddings_out_round)

            return out_message, embeddings_out, semantic_search_file_name

        return out_message, embeddings_out, None
    else:
        # Just return existing embeddings if already exist
        embeddings_out = embeddings_state
    
    print(out_message)

    return out_message, embeddings_out, None#, None

def jina_simple_retrieval_deprecated(query_str:str, vectorstore, docs, orig_df_col:str, k_val:int, out_passages:int,
                           vec_score_cut_off:float, vec_weight:float, in_join_file, in_join_column = None, search_df_join_column = None, device = torch_device, embeddings = embeddings_model, progress=gr.Progress(track_tqdm=True)): # ,vectorstore, embeddings

    # print("vectorstore loaded: ", vectorstore)
    progress(0, desc = "Conducting semantic search")

    print("Searching")

    # Convert it to a PyTorch tensor and transfer to GPU
    vectorstore_tensor = tensor(vectorstore).to(device)

    # Load the sentence transformer model and move it to GPU
    embeddings = embeddings.to(device)

    # Encode the query using the sentence transformer and convert to a PyTorch tensor
    query = embeddings.encode(query_str)
    query_tensor = tensor(query).to(device)

    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)  # Reshape to 2D with one row

    # Normalize the query tensor and vectorstore tensor
    query_norm = query_tensor / query_tensor.norm(dim=1, keepdim=True)
    vectorstore_norm = vectorstore_tensor / vectorstore_tensor.norm(dim=1, keepdim=True)

    # Calculate cosine similarities (batch processing)
    cosine_similarities = mm(query_norm, vectorstore_norm.T)

    # Flatten the tensor to a 1D array
    cosine_similarities = cosine_similarities.flatten()

    # Convert to a NumPy array if it's still a PyTorch tensor
    cosine_similarities = cosine_similarities.cpu().numpy()

    # Create a Pandas Series
    cosine_similarities_series = pd.Series(cosine_similarities)

    # Pull out relevent info from docs
    page_contents = [doc.page_content for doc in docs]
    page_meta = [doc.metadata for doc in docs]
    ids_range = range(0,len(page_contents)) 
    ids = [str(element) for element in ids_range]

    df_docs = pd.DataFrame(data={"ids": ids,
                                "documents": page_contents,
                                    "metadatas":page_meta,
                                    "distances":cosine_similarities_series}).sort_values("distances", ascending=False).iloc[0:k_val,:]

    
    results_df_out = process_data_from_scores_df(df_docs, in_join_file, out_passages, vec_score_cut_off, vec_weight, orig_df_col, in_join_column, search_df_join_column)

    print("Search complete")

    # If nothing found, return error message
    if results_df_out.empty:
        return 'No result found!', None
    
    query_str_file = query_str.replace(" ", "_")

    results_df_name = "semantic_search_result_" + today_rev + "_" +  query_str_file + ".xlsx"

    print("Saving search output to file")
    progress(0.7, desc = "Saving search output to file")

    results_df_out.to_excel(results_df_name, index= None)
    results_first_text = results_df_out.iloc[0, 1]

    print("Returning results")

    return results_first_text, results_df_name

# Deprecated Chroma functions - kept just in case needed in future.
# Chroma support is currently deprecated
# Import Chroma and instantiate a client. The default Chroma client is ephemeral, meaning it will not save to disk.
#import chromadb
#from chromadb.config import Settings
#from typing_extensions import Protocol
#from chromadb import Documents, EmbeddingFunction, Embeddings

# Remove Chroma database file. If it exists as it can cause issues
#chromadb_file = "chroma.sqlite3"

#if os.path.isfile(chromadb_file):
#    os.remove(chromadb_file)


def docs_to_chroma_save_deprecated(docs_out, embeddings = embeddings_model, progress=gr.Progress()):
    '''
    Takes a Langchain document class and saves it into a Chroma sqlite file. Not currently used.
    '''

    print(f"> Total split documents: {len(docs_out)}")

    #print(docs_out)

    page_contents = [doc.page_content for doc in docs_out]
    page_meta = [doc.metadata for doc in docs_out]
    ids_range = range(0,len(page_contents)) 
    ids = [str(element) for element in ids_range]

    tic = time.perf_counter()
    #embeddings_list = []
    #for page in progress.tqdm(page_contents, desc = "Preparing search index", unit = "rows"):
    #    embeddings_list.append(embeddings.encode(sentences=page, max_length=1024).tolist())

    embeddings_list = embeddings.encode(sentences=page_contents, max_length=256, show_progress_bar = True, batch_size = 32).tolist() # For Jina embeddings
    #embeddings_list = embeddings.encode(sentences=page_contents, normalize_embeddings=True).tolist() # For BGE embeddings
    #embeddings_list = embeddings.encode(sentences=page_contents).tolist() # For minilm

    toc = time.perf_counter()
    time_out = f"The embedding took {toc - tic:0.1f} seconds"

    #pd.Series(embeddings_list).to_csv("embeddings_out.csv")

    # Jina tiny
    # This takes about 300 seconds for 240,000 records = 800 / second, 1024 max length
    # For 50k records:
    # 61 seconds at 1024 max length
    # 55 seconds at 512 max length
    # 43 seconds at 256 max length
    # 31 seconds at 128 max length

    # The embedding took 1372.5 seconds at 256 max length for 655,020 case notes

    # BGE small
    # 96 seconds for 50k records at 512 length

    # all-MiniLM-L6-v2
    # 42.5 seconds at (256?) max length

    # paraphrase-MiniLM-L3-v2
    # 22 seconds for 128 max length


    print(time_out)

    chroma_tic = time.perf_counter()

    # Create a new Chroma collection to store the documents and metadata. We don't need to specify an embedding fuction, and the default will be used.
    client = chromadb.PersistentClient(path="./last_year", settings=Settings(
    anonymized_telemetry=False))

    try:
        print("Deleting existing collection.")
        #collection = client.get_collection(name="my_collection")
        client.delete_collection(name="my_collection")
        print("Creating new collection.")
        collection = client.create_collection(name="my_collection")
    except: 
        print("Creating new collection.")
        collection = client.create_collection(name="my_collection")

    # Match batch size is about 40,000, so add that amount in a loop
    def create_batch_ranges(in_list, batch_size=40000):
        total_rows = len(in_list)
        ranges = []
        
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            ranges.append(range(start, end))
        
        return ranges

    batch_ranges = create_batch_ranges(embeddings_list)
    print(batch_ranges)

    for row_range in progress.tqdm(batch_ranges, desc = "Creating vector database", unit = "batches of 40,000 rows"):
    
        collection.add(
        documents = page_contents[row_range[0]:row_range[-1]],
        embeddings = embeddings_list[row_range[0]:row_range[-1]],
        metadatas = page_meta[row_range[0]:row_range[-1]],
        ids = ids[row_range[0]:row_range[-1]])
        #print("Here")
        
    # print(collection.count())
    

    #chatf.vectorstore = vectorstore_func

    chroma_toc = time.perf_counter()

    chroma_time_out = f"Loading to Chroma db took {chroma_toc - chroma_tic:0.1f} seconds"
    print(chroma_time_out)

    out_message = "Document processing complete"

    return out_message, collection

def chroma_retrieval_deprecated(query_str:str, vectorstore, docs, orig_df_col:str, k_val:int, out_passages:int,
                           vec_score_cut_off:float, vec_weight:float, in_join_file = None, in_join_column = None, search_df_join_column = None, embeddings = embeddings_model): # ,vectorstore, embeddings

            query = embeddings.encode(query_str).tolist()

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
            
            results_df_out = process_data_from_scores_df(df_docs, in_join_file, out_passages, vec_score_cut_off, vec_weight, orig_df_col, in_join_column, search_df_join_column)

            results_df_name = output_folder + "semantic_search_result.csv"
            results_df_out.to_csv(results_df_name, index= None)
            results_first_text = results_df_out[orig_df_col].iloc[0]

            return results_first_text, results_df_name

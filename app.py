from typing import Type
import gradio as gr
import pandas as pd
import numpy as np
import os
PandasDataFrame = Type[pd.DataFrame]

from search_funcs.bm25_functions import prepare_bm25_input_data, prepare_bm25, bm25_search
from search_funcs.semantic_ingest_functions import csv_excel_text_to_docs
from search_funcs.semantic_functions import load_embedding_model, docs_to_embed_np_array, bge_semantic_search
from search_funcs.helper_functions import display_info, initial_data_load, put_columns_in_join_df, get_connection_params, output_folder, get_or_create_env_var # Not currently used: get_temp_folder_path, empty_folder, 
from search_funcs.spacy_search_funcs import spacy_fuzzy_search
from search_funcs.aws_functions import load_data_from_aws
from search_funcs.auth import authenticate_user

# Attempt to delete temporary files generated by previous use of the app (as the files can be very big!). Only setup to work for local runs in Windows (not used at the moment).
# temp_folder_path = get_temp_folder_path()
# empty_folder(temp_folder_path)

## Gradio app - BM25 search
app = gr.Blocks(theme = gr.themes.Base()) # , css="theme.css"

with app:
    print("Please don't close this window! Open the below link in the web browser of your choice.")

    # BM25 state objects
    orig_keyword_data_state = gr.State(pd.DataFrame()) # Original data that is not changed #gr.Dataframe(pd.DataFrame(),visible=False) #gr.State(pd.DataFrame())
    prepared_keyword_data_state = gr.State(pd.DataFrame()) # Data frame the contains modified data #gr.Dataframe(pd.DataFrame(),visible=False) #gr.State(pd.DataFrame())
    tokenised_prepared_keyword_data_state = gr.State([]) # Data that has been prepared for search (tokenised) #gr.Dataframe(np.array([]), type="array", visible=False) #gr.State([])
    bm25_search_index_state = gr.State()

    # Semantic search state objects
    orig_semantic_data_state = gr.State(pd.DataFrame()) #gr.Dataframe(pd.DataFrame(),visible=False) # gr.State(pd.DataFrame())
    semantic_data_state = gr.State(pd.DataFrame()) #gr.Dataframe(pd.DataFrame(),visible=False) # gr.State(pd.DataFrame())
    semantic_input_document_format = gr.State([])

    embeddings_model_name_state = gr.State("sentence-transformers/all-MiniLM-L6-v2")#"BAAI/bge-small-en-v1.5")
    embeddings_model_loc_state = gr.State("minilm/")#"bge/")
    embeddings_state = gr.State(np.array([])) #gr.Dataframe(np.array([]), type="numpy", visible=False) #gr.State(np.array([])) # globals()["embeddings"]
    embeddings_model_state = gr.State()
    torch_device_state = gr.State("cpu")
    semantic_k_val = gr.Number(9999, visible=False)

    # State objects for app in general
    session_hash_state = gr.State("")
    s3_output_folder_state = gr.State("")
    join_data_state = gr.State(pd.DataFrame()) #gr.Dataframe(pd.DataFrame(), visible=False) #gr.State(pd.DataFrame())
    output_file_state = gr.State([]) #gr.Dataframe(type="array", visible=False) #gr.State([])

    # Informational state objects
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
    # Data text search
    Search through long-form text fields in your tabular data. Either for exact, specific terms (Keyword search), or thematic, 'fuzzy' search (Semantic search). More instructions are provided in the relevant tabs below.
    """)
    
    with gr.Tab(label="Keyword search"):
        gr.Markdown(
    """
    **Exact term keyword search**
    
    BM25 based keyword search over tabular open text data. 1. Load in data file (.csv, .xlsx, or .parquet), and if you have searched with this dataset and saved the processing outputs, the '...search_index.pkl.gz' in the same folder to save loading time. 2. Select the field in your data to search. A field with the suffix '_cleaned' means that html tags have been removed. 3. Wait for the data file to be prepared for search. 4. Enter the search term in the relevant box below and press Enter/click on 'Search text'. 4. Your search results will be saved in an .xlsx file and will be presented in the 'File output' area below.
    """)
        with gr.Row():
            current_source = gr.Textbox(label="Current data source(s)", value="None")

        with gr.Accordion(label = "Load in data", open=True):
            in_bm25_file = gr.File(label="Upload data for keyword search", file_count= 'multiple', file_types =['.parquet', '.csv', '.pkl', '.pkl.gz', '.zip'])
            with gr.Row():
                in_bm25_column = gr.Dropdown(label="Enter the name of the text column in the data file to search") 
                load_bm25_data_button = gr.Button(value="Load data")
                 
            with gr.Row():
                load_finished_message = gr.Textbox(label="Load progress", scale = 2)

        with gr.Accordion(label = "Search data", open=True):
            keyword_query = gr.Textbox(label="Enter your search term")
            with gr.Row():
                keyword_search_button = gr.Button(value="Keyword search", variant="primary", scale=1)
                fuzzy_search_button = gr.Button(value="Fuzzy search (slow, < 10k rows)", variant="secondary", scale = 0)
            with gr.Row():
                output_single_text = gr.Textbox(label="Top result")
                output_file = gr.File(label="File output")

    with gr.Tab("Semantic search"):
        gr.Markdown(
    """
    **Thematic/semantic search**

    This search type enables you to search for general terms (e.g. happiness, nature) and the search will pick out text passages that are most semantically similar to them. 1. Load in data file (ideally a file with '_cleaned' at the end of the name, a pkl.gz file), with (optionally) the 'embed... .npz' file in the same folder to save loading time. 2. Select the field in your data to search. 3. Wait for the data file to be prepared for search. 4. Enter the search term in the 'Enter semantic search query here' box below and press Enter/click on 'Start semantic search'. 4. Your search results will be saved in a csv file and will be presented in the 'File output' area below.
    """)       

        with gr.Row():
            current_source_semantic = gr.Textbox(label="Current data source(s)", value="None")

        with gr.Accordion("Load in data", open = True):
            in_semantic_file = gr.File(label="Upload data file for semantic search", file_count= 'multiple', file_types = ['.parquet', '.csv', '.npy', '.npz', '.pkl', '.pkl.gz', '.zip'])
            
            with gr.Row():
                in_semantic_column = gr.Dropdown(label="Enter the name of the text column in the data file to search")
                load_semantic_data_button = gr.Button(value="Load data", variant="secondary")
                
            semantic_load_progress = gr.Textbox(label="Load progress")
        
        with gr.Accordion(label="Semantic search options", open = False):
            semantic_min_distance = gr.Slider(label = "Minimum distance score for search result to be included", value = 0.2, minimum=0, maximum=0.95, step=0.01)
        
        semantic_query = gr.Textbox(label="Enter semantic search query here")
        semantic_submit = gr.Button(value="Start semantic search", variant="primary")

        with gr.Row():
            semantic_output_single_text = gr.Textbox(label="Top result")
            semantic_output_file = gr.File(label="File output")
            
    with gr.Tab(label="Advanced options"):
        with gr.Accordion(label="Data load / save options", open = True):
            with gr.Row():
                in_clean_data = gr.Dropdown(label = "Clean text during load (remove html tags). For large files this may take some time!", value="No", choices=["Yes", "No"])
                return_intermediate_files = gr.Dropdown(label = "Return intermediate processing files from file preparation. Files can be loaded in to save processing time in future.", value="No", choices=["Yes", "No"])
                embeddings_compress = gr.Dropdown(label = "Round embeddings to int8 precision for smaller files with less accuracy.", value="Yes", choices=["Yes", "No"])
            #save_clean_data_button = gr.Button(value = "Save loaded data to file", scale = 1)
        with gr.Accordion(label="Keyword search options", open = False):
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
        with gr.Accordion(label="Fuzzy search options", open = False):
                no_spelling_mistakes = gr.Slider(label = "Number of spelling mistakes allowed in fuzzy search", value = 1, minimum=1, maximum=4, step=1)
        
        with gr.Accordion(label = "Join on additional dataframes to results", open = False):
            in_join_file = gr.File(label="Upload your data to join here")
            in_join_message = gr.Textbox(label="Join file load progress")
            in_join_column = gr.Dropdown(label="Column to join in new data frame")
            search_df_join_column = gr.Dropdown(label="Column to join in search data frame")

        with gr.Accordion(label = "AWS data access", open = False):
            aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
            with gr.Row():
                in_aws_keyword_file = gr.Dropdown(label="Choose keyword file to load from AWS (only valid for API Gateway app)", choices=["None", "Bioasq - Biomedical example data - keyword search"])
                load_aws_keyword_data_button = gr.Button(value="Load keyword data from AWS", variant="secondary")
            with gr.Row():
                in_aws_semantic_file = gr.Dropdown(label="Choose semantic file to load from AWS (only valid for API Gateway app)", choices=["None", "Bioasq - Biomedical example data - semantic search"])
                load_aws_semantic_data_button = gr.Button(value="Load semantic data from AWS", variant="secondary")
            
            out_aws_data_message = gr.Textbox(label="AWS data load progress")

    # Changing search parameters button
    in_search_param_button.click(fn=prepare_bm25, inputs=[tokenised_prepared_keyword_data_state, in_bm25_file, in_bm25_column, bm25_search_index_state, return_intermediate_files, in_k1, in_b, in_alpha], outputs=[load_finished_message])
               
    # ---
    in_k1_button.click(display_info, inputs=in_k1_info)
    in_b_button.click(display_info, inputs=in_b_info)
    in_alpha_button.click(display_info, inputs=in_alpha_info)
    in_no_search_results_button.click(display_info, inputs=in_no_search_info)

    ### Loading AWS data ###
    load_aws_keyword_data_button.click(fn=load_data_from_aws, inputs=[in_aws_keyword_file, aws_password_box], outputs=[in_bm25_file, out_aws_data_message])
    load_aws_semantic_data_button.click(fn=load_data_from_aws, inputs=[in_aws_semantic_file, aws_password_box], outputs=[in_semantic_file, out_aws_data_message])
    
    ### BM25 SEARCH ###
    # Update dropdowns upon initial file load
    in_bm25_file.upload(initial_data_load, inputs=[in_bm25_file], outputs=[in_bm25_column, search_df_join_column, prepared_keyword_data_state, orig_keyword_data_state, bm25_search_index_state, embeddings_state, tokenised_prepared_keyword_data_state, load_finished_message, current_source, in_bm25_file], api_name="keyword_data_load")
    in_join_file.change(put_columns_in_join_df, inputs=[in_join_file], outputs=[in_join_column, join_data_state, in_join_message])
 
    # Load in BM25 data
    load_bm25_data_button.click(fn=prepare_bm25_input_data, inputs=[in_bm25_file, in_bm25_column, prepared_keyword_data_state, tokenised_prepared_keyword_data_state, in_clean_data, return_intermediate_files], outputs=[tokenised_prepared_keyword_data_state, load_finished_message, prepared_keyword_data_state, output_file, output_file, in_bm25_column], api_name="load_keyword").\
    then(fn=prepare_bm25, inputs=[tokenised_prepared_keyword_data_state, in_bm25_file, in_bm25_column, bm25_search_index_state, in_clean_data, return_intermediate_files, in_k1, in_b, in_alpha], outputs=[load_finished_message, output_file, bm25_search_index_state, tokenised_prepared_keyword_data_state], api_name="prepare_keyword") # keyword_data_list_state
    
    # BM25 search functions on click or enter
    keyword_search_button.click(fn=bm25_search, inputs=[keyword_query, in_no_search_results, orig_keyword_data_state, prepared_keyword_data_state, in_bm25_column, join_data_state, in_clean_data, bm25_search_index_state, tokenised_prepared_keyword_data_state, in_join_column, search_df_join_column, in_k1, in_b, in_alpha], outputs=[output_single_text, output_file], api_name="keyword_search")
    keyword_query.submit(fn=bm25_search, inputs=[keyword_query, in_no_search_results, orig_keyword_data_state, prepared_keyword_data_state, in_bm25_column, join_data_state, in_clean_data, bm25_search_index_state, tokenised_prepared_keyword_data_state, in_join_column, search_df_join_column, in_k1, in_b, in_alpha], outputs=[output_single_text, output_file])

    # Fuzzy search functions on click
    fuzzy_search_button.click(fn=spacy_fuzzy_search, inputs=[keyword_query, tokenised_prepared_keyword_data_state, prepared_keyword_data_state, in_bm25_column, join_data_state, search_df_join_column, in_join_column, no_spelling_mistakes], outputs=[output_single_text, output_file], api_name="fuzzy_search")
    
    ### SEMANTIC SEARCH ###

    # Load in a csv/excel file for semantic search
    in_semantic_file.upload(initial_data_load, inputs=[in_semantic_file], outputs=[in_semantic_column,  search_df_join_column,  semantic_data_state, orig_semantic_data_state, bm25_search_index_state, embeddings_state, tokenised_prepared_keyword_data_state, semantic_load_progress, current_source_semantic, in_semantic_file], api_name="semantic_data_load")
    
    load_semantic_data_button.click(
        csv_excel_text_to_docs, inputs=[semantic_data_state, in_semantic_file, in_semantic_column, in_clean_data, return_intermediate_files], outputs=[semantic_input_document_format, semantic_load_progress, output_file_state], api_name="convert_texts_to_documents").\
        then(docs_to_embed_np_array, inputs=[semantic_input_document_format, in_semantic_file, output_file_state, in_clean_data, embeddings_state, embeddings_model_name_state, embeddings_model_loc_state, return_intermediate_files, embeddings_compress], outputs=[semantic_load_progress, embeddings_state, semantic_output_file, output_file_state, embeddings_model_state], api_name="embed_documents")

    # Semantic search query
    semantic_submit.click(bge_semantic_search, inputs=[semantic_query, embeddings_state, semantic_input_document_format, semantic_k_val, semantic_min_distance, embeddings_model_state, embeddings_model_name_state, embeddings_compress, join_data_state, in_join_column, search_df_join_column], outputs=[semantic_output_single_text, semantic_output_file], api_name="semantic_search")
    semantic_query.submit(bge_semantic_search, inputs=[semantic_query, embeddings_state, semantic_input_document_format, semantic_k_val, semantic_min_distance, embeddings_model_state, embeddings_model_name_state, embeddings_compress, join_data_state, in_join_column, search_df_join_column], outputs=[semantic_output_single_text, semantic_output_file])

    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state])

COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')
print(f'The value of COGNITO_AUTH is {COGNITO_AUTH}')

if __name__ == "__main__":

    if os.environ['COGNITO_AUTH'] == "1":
        app.queue().launch(show_error=True, auth=authenticate_user)
    else:
        app.queue().launch(show_error=True, inbrowser=True)
    
# Running on local server with https: https://discuss.huggingface.co/t/how-to-run-gradio-with-0-0-0-0-and-https/38003 or https://dev.to/rajshirolkar/fastapi-over-https-for-development-on-windows-2p7d # Need to download OpenSSL and create own keys 
# app.queue().launch(ssl_verify=False, share=False, debug=False, server_name="0.0.0.0",server_port=443,
#                      ssl_certfile="cert.pem", ssl_keyfile="key.pem") # port 443 for https. Certificates currently not valid
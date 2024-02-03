import spacy
from spacy.matcher import Matcher
import numpy as np
import gradio as gr
import pandas as pd
from typing import List, Type
from datetime import datetime

PandasDataFrame = Type[pd.DataFrame]

today_rev = datetime.now().strftime("%Y%m%d")

nlp = spacy.load("en_core_web_sm")

string_query = "knife attack run fast"
df_list = ["Last week someone was grievously injured in a knife attack on Exmoor road. Running away. They ran as fast as possible. I run.","This is the 3rd knifing in the area in as many weeks; knives everywhere.", "attacks of this kind have been increasing for years. Knife attack or knife attack.", "Nothing happened here"]


def spacy_fuzzy_search(string_query:str, df_list: List[str], original_data: PandasDataFrame, text_column:str, in_join_file: PandasDataFrame, search_df_join_column:str, in_join_column:str, no_spelling_mistakes:int = 1, progress=gr.Progress(track_tqdm=True)):
    ''' Conduct fuzzy match on a list of data.'''

    query = nlp(string_query)
    tokenised_query = [token.text for token in query]
    print(tokenised_query)

    spelling_mistakes_fuzzy_pattern = "FUZZY" + str(no_spelling_mistakes)

    # %%
    if len(tokenised_query) > 1:
        pattern_lemma = [{"LEMMA": {"IN": tokenised_query}}]
        pattern_fuzz = [{"TEXT": {spelling_mistakes_fuzzy_pattern: {"IN": tokenised_query}}}]
    else:
        pattern_lemma = [{"LEMMA": tokenised_query[0]}]
        pattern_fuzz = [{"TEXT": {spelling_mistakes_fuzzy_pattern: tokenised_query[0]}}]

   
    # %%
    matcher = Matcher(nlp.vocab)
    
    # %%
    matcher.add(string_query, [pattern_fuzz])
    matcher.add(string_query, [pattern_lemma])

    # %%
    batch_size = 256
    docs = nlp.pipe(df_list, batch_size=batch_size)

    # %%
    all_matches = []   

    # Get number of matches per doc
    for doc in progress.tqdm(docs, desc = "Searching text", unit = "rows"):
        matches = matcher(doc)
        match_count = len(matches)
        all_matches.append(match_count)

    print("Search complete")

    ## Get document lengths
    lengths = []
    for element in df_list:
        lengths.append(len(element))
        
    # Score is number of matches divided by length of document
    match_scores = (np.array(all_matches)/np.array(lengths)).tolist()

    # Prepare results and export
    results_df = pd.DataFrame(data={"index": list(range(len(df_list))),
                                    "search_text": df_list,
                                    "search_score_abs": match_scores})
    results_df['search_score_abs'] = abs(round(results_df['search_score_abs']*100, 2))
    results_df_out = results_df[['index', 'search_text', 'search_score_abs']].merge(original_data,left_on="index", right_index=True, how="left")

    # Keep only results with at least one match
    results_df_out = results_df_out.loc[results_df["search_score_abs"] > 0, :]

    # Join on additional files
    if not in_join_file.empty:
        progress(0.5, desc = "Joining on additional data file")
        join_df = in_join_file
        join_df[in_join_column] = join_df[in_join_column].astype(str).str.replace("\.0$","", regex=True)
        results_df_out[search_df_join_column] = results_df_out[search_df_join_column].astype(str).str.replace("\.0$","", regex=True)

        # Duplicates dropped so as not to expand out dataframe
        join_df = join_df.drop_duplicates(in_join_column)

        results_df_out = results_df_out.merge(join_df,left_on=search_df_join_column, right_on=in_join_column, how="left", suffixes=('','_y'))#.drop(in_join_column, axis=1)

    # Reorder results by score
    results_df_out = results_df_out.sort_values('search_score_abs', ascending=False)

    # Out file
    query_str_file = ("_").join(tokenised_query)
    results_df_name = "keyword_search_result_" + today_rev + "_" +  query_str_file + ".xlsx"

    print("Saving search file output")
    progress(0.7, desc = "Saving search output to file")

    results_df_out.to_excel(results_df_name, index= None)
    results_first_text = results_df_out[text_column].iloc[0]

    print("Returning results")

    return results_first_text, results_df_name
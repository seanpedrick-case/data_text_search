---
title: Data text search
emoji: 🔍
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
license: apache-2.0
---

Keyword search over your data. This is an adaptation of fast_bm25 (https://github.com/Inspirateur/Fast-BM25) to search over tabular data with a Gradio UI interface.

# Guide

1. Load in your tabular data file (.csv, .parquet, .xlsx - first sheet).
2. Wait a few seconds for the file to upload, then in the dropdown menu below 'Enter the name of the text column...' choose the column from the data file that you want to search.
3. Hit 'Load data'. The 'Load progress' text box will let you know when the file is ready.
4. In the 'Enter your search term' area below this, type in the key words you want to find in your text. Note that if the term is not spelled exactly as it is found in the text, it will not be found!
5. Hit search text. You may have to wait depending on the size of the data you are searching.
6. You will receive back 1. the top search result and 2. a csv of the search results found in the text ordered by relevance, joined onto the original columns from your data source.

# Advanced options
The search should perform well with default options, so you shouldn't need to change things here.

## Data load / save options
Toggle 'Clean text during load...' to true if you want to remove html tags and lemmatise the text, i.e. remove the ends of words to retain the core of the word e.g. searched or searches becomes search. Early testing suggests that cleaning takes some time, and does not seem to improve quality of search results.

## Search options
Here are a few options to modify the BM25 search parameters. If you want more information on what each parameter does, click the relevant info button to the right of the sliders.

## Join on additional dataframes to results
I was asked to include a feature to join on additional data to the search results. This could be useful for example if you have tabular text data associated with a person ID, and after searching you would like to join on information associated with this person to aid with post-search filtering/analysis.

To do this:
1. Load in the tabular data you want to join in the box (.csv, .parquet, .xlsx - first sheet). 
2. Then choose the field that you want to join onto the results sheet, and the matching field from the data you are searching with.
3. Next time you do a search on the first tab, the new data should be joined onto your output file.

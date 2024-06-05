import collections
import heapq
import math
import pickle
import sys
import gzip
import time
import pandas as pd
from numpy import inf
import gradio as gr

from datetime import datetime

today_rev = datetime.now().strftime("%Y%m%d")

from search_funcs.clean_funcs import initial_clean # get_lemma_tokens, stem_sentence
from search_funcs.helper_functions import get_file_path_end_with_ext, get_file_path_end, create_highlighted_excel_wb, ensure_output_folder_exists, output_folder

# Load the SpaCy model
from spacy.cli.download import download
import spacy
spacy.prefer_gpu()

#os.system("python -m spacy download en_core_web_sm")
try:
	import en_core_web_sm
	nlp = en_core_web_sm.load()
	print("Successfully imported spaCy model")
    #nlp = spacy.load("en_core_web_sm")
    #print(nlp._path)
except:
	download("en_core_web_sm")
	nlp = spacy.load("en_core_web_sm")
	print("Successfully imported spaCy model")
    #print(nlp._path)

# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

PARAM_K1 = 1.5
PARAM_B = 0.75
IDF_CUTOFF = -inf

# Class built off https://github.com/Inspirateur/Fast-BM25

class BM25:
	"""Fast Implementation of Best Matching 25 ranking function.

	Attributes
	----------
	t2d : <token: <doc, freq>>
		Dictionary with terms frequencies for each document in `corpus`.
	idf: <token, idf score>
		Pre computed IDF score for every term.
	doc_len : list of int
		List of document lengths.
	avgdl : float
		Average length of document in `corpus`.
	"""
	def __init__(self, corpus, k1=PARAM_K1, b=PARAM_B, alpha=IDF_CUTOFF):
		"""
		Parameters
		----------
		corpus : list of list of str
			Given corpus.
		k1 : float
			Constant used for influencing the term frequency saturation. After saturation is reached, additional
			presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
			that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
			the type of documents or queries.
		b : float
			Constant used for influencing the effects of different document lengths relative to average document length.
			When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
			[1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
			depends on factors such as the type of documents or queries.
		alpha: float
			IDF cutoff, terms with a lower idf score than alpha will be dropped. A higher alpha will lower the accuracy
			of BM25 but increase performance
		"""
		self.k1 = k1
		self.b = b
		self.alpha = alpha
		self.corpus = corpus

		self.avgdl = 0
		self.t2d = {}
		self.idf = {}
		self.doc_len = []
		if corpus:
			self._initialize(corpus)

	@property
	def corpus_size(self):
		return len(self.doc_len)

	def _initialize(self, corpus, progress=gr.Progress()):
		"""Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
		i = 0
		for document in progress.tqdm(corpus, desc = "Preparing search index", unit = "rows"):
			self.doc_len.append(len(document))

			for word in document:
				if word not in self.t2d:
					self.t2d[word] = {}
				if i not in self.t2d[word]:
					self.t2d[word][i] = 0
				self.t2d[word][i] += 1
			i += 1

		self.avgdl = sum(self.doc_len)/len(self.doc_len)
		to_delete = []
		for word, docs in self.t2d.items():
			idf = math.log(self.corpus_size - len(docs) + 0.5) - math.log(len(docs) + 0.5)
			# only store the idf score if it's above the threshold
			if idf > self.alpha:
				self.idf[word] = idf
			else:
				to_delete.append(word)
		print(f"Dropping {len(to_delete)} terms")
		for word in to_delete:
			del self.t2d[word]

		if len(self.idf) == 0:
			print("Alpha value too high - all words removed from dataset.")
			self.average_idf = 0

		else:
			self.average_idf = sum(self.idf.values())/len(self.idf)

		if self.average_idf < 0:
			print(
				f'Average inverse document frequency is less than zero. Your corpus of {self.corpus_size} documents'
				' is either too small or it does not originate from natural text. BM25 may produce'
				' unintuitive results.',
				file=sys.stderr
			)

	def get_top_n(self, query, documents, n=5):
		"""
		Retrieve the top n documents for the query.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		list
			The top n documents
		"""
		assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
		scores = collections.defaultdict(float)
		for token in query:
			if token in self.t2d:
				for index, freq in self.t2d[token].items():
					denom_cst = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
					scores[index] += self.idf[token]*freq*(self.k1 + 1)/(freq + denom_cst)

		return [documents[i] for i in heapq.nlargest(n, scores.keys(), key=scores.__getitem__)]
	

	def get_top_n_with_score(self, query, documents, n=5):
		"""
		Retrieve the top n documents for the query along with their scores.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		list
			The top n documents along with their scores and row indices in the format (index, document, score)
		"""
		assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
		scores = collections.defaultdict(float)
		for token in query:
			if token in self.t2d:
				for index, freq in self.t2d[token].items():
					denom_cst = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
					scores[index] += self.idf[token] * freq * (self.k1 + 1) / (freq + denom_cst)

		top_n_indices = heapq.nlargest(n, scores.keys(), key=scores.__getitem__)
		return [(i, documents[i], scores[i]) for i in top_n_indices]
	
	def extract_documents_and_scores(self, query, documents, n=5):
		"""
		Extract top n documents and their scores into separate lists.

		Parameters
		----------
		query: list of str
			The tokenized query
		documents: list
			The documents to return from
		n: int
			The number of documents to return

		Returns
		-------
		tuple: (list, list)
			The first list contains the top n documents and the second list contains their scores.
		"""
		results = self.get_top_n_with_score(query, documents, n)
		try:
			indices, docs, scores = zip(*results)
		except:
			print("No search results returned")
			return [], [], []
		return list(indices), docs, list(scores)

	def save(self, filename):
		with open(f"{output_folder}{filename}.pkl", "wb") as fsave:
			pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		with open(f"{output_folder}{filename}.pkl", "rb") as fsave:
			return pickle.load(fsave)

# These following functions are my own work

def prepare_bm25_input_data(in_file, text_column, data_state, tokenised_state, clean="No",  return_intermediate_files = "No", progress=gr.Progress(track_tqdm=True)):
	#print(in_file)
	ensure_output_folder_exists(output_folder)

	if not in_file:
		print("No input file found. Please load in at least one file.")
		return None, "No input file found. Please load in at least one file.", data_state, None, None, [], gr.Dropdown(allow_custom_value=True, value=text_column, choices=data_state.columns.to_list())

	progress(0, desc = "Loading in data")
	file_list = [string.name for string in in_file]

	#print(file_list)

	data_file_names = [string for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower() and "gz" not in string.lower()]

	if not data_file_names:
		return None, "Please load in at least one csv/Excel/parquet data file.", data_state, None, None, [], gr.Dropdown(allow_custom_value=True, value=text_column, choices=data_state.columns.to_list())

	if not text_column:
		return None, "Please enter a column name to search.", data_state, None, None,[], gr.Dropdown(allow_custom_value=True, value=text_column, choices=data_state.columns.to_list())

	data_file_name = data_file_names[0]

	df = data_state #read_file(data_file_name)
	data_file_out_name = get_file_path_end_with_ext(data_file_name)
	data_file_out_name_no_ext = get_file_path_end(data_file_name)

	## Load in pre-tokenised corpus if exists
	tokenised_df = pd.DataFrame()

	tokenised_file_names = [string for string in file_list if "tokenised" in string.lower()]
	search_index_file_names = [string for string in file_list if "gz" in string.lower()]

	df[text_column] = df[text_column].astype(str).str.lower()

	if "copy_of_case_note_id" in df.columns:
		print("copy column found")
		df.loc[~df["copy_of_case_note_id"].isna(), text_column] = ""

	if search_index_file_names:
		corpus = list(df[text_column])
		message = "Tokenisation skipped - loading search index from file."
		print(message)
		return corpus, message, df, None, None, [], gr.Dropdown(allow_custom_value=True, value=text_column, choices=data_state.columns.to_list())

	
	if clean == "Yes":
		progress(0.1, desc = "Cleaning data")
		clean_tic = time.perf_counter()
		print("Starting data clean.")

		#df = df.drop_duplicates(text_column)
		df_list = list(df[text_column])
		df_list = initial_clean(df_list)

		# Save to file if you have cleaned the data
		out_file_name, text_column, df  = save_prepared_bm25_data(data_file_name, df_list, df, text_column)
	
		clean_toc = time.perf_counter()
		clean_time_out = f"Cleaning the text took {clean_toc - clean_tic:0.1f} seconds."
		print(clean_time_out)

	else:
		# Don't clean or save file to disk
		df_list = list(df[text_column])
		print("No data cleaning performed")
		out_file_name = None
		
	# Tokenise data. If tokenised df already exists, no need to do anything
	
	progress(0.4, desc = "Tokenising text")

	if tokenised_state:
		tokenised_df = tokenised_state
		corpus = tokenised_df.iloc[:,0].tolist()
		print("Tokenised data loaded from file")
		#print("Corpus is: ", corpus[0:5])

	else:
		tokeniser_tic = time.perf_counter()
		corpus = []
		batch_size = 256
		for doc in tokenizer.pipe(progress.tqdm(df_list, desc = "Tokenising text", unit = "rows"), batch_size=batch_size):
			corpus.append([token.text for token in doc])

		tokeniser_toc = time.perf_counter()
		tokenizer_time_out = f"Tokenising the text took {tokeniser_toc - tokeniser_tic:0.1f} seconds."
		print(tokenizer_time_out)

	if len(df_list) >= 20:
		message = "Data loaded"
	else:
		message = "Data loaded. Warning: dataset may be too short to get consistent search results."

	if return_intermediate_files == "Yes":

		if clean == "Yes":
			tokenised_data_file_name = output_folder + data_file_out_name_no_ext + "_cleaned_tokenised.parquet"
		else:
			tokenised_data_file_name = output_folder + data_file_out_name_no_ext + "_tokenised.parquet"

		pd.DataFrame(data={"Corpus":corpus}).to_parquet(tokenised_data_file_name)

		return corpus, message, df, out_file_name, tokenised_data_file_name, df_list, gr.Dropdown(allow_custom_value=True, value=text_column, choices=data_state.columns.to_list())

	return corpus, message, df, out_file_name, None, df_list, gr.Dropdown(allow_custom_value=True, value=text_column, choices=data_state.columns.to_list())

def save_prepared_bm25_data(in_file_name, prepared_text_list, in_df, in_bm25_column, progress=gr.Progress(track_tqdm=True)):

	ensure_output_folder_exists(output_folder)

	# Check if the list and the dataframe have the same length
	if len(prepared_text_list) != len(in_df):
		raise ValueError("The length of 'prepared_text_list' and 'in_df' must match.")

	file_end = ".parquet"

	file_name = output_folder + get_file_path_end(in_file_name) + "_cleaned" + file_end

	new_text_column = in_bm25_column + "_cleaned"
	prepared_text_df = pd.DataFrame(data={new_text_column:prepared_text_list})

	# Drop original column from input file to reduce file size
	in_df = in_df.drop(in_bm25_column, axis = 1)

	prepared_df = pd.concat([in_df, prepared_text_df], axis = 1)

	if file_end == ".csv":
		prepared_df.to_csv(file_name)
	elif file_end == ".parquet":
		prepared_df.to_parquet(file_name)
	else: file_name = None

	return file_name, new_text_column, prepared_df

def prepare_bm25(corpus, in_file, text_column, search_index, clean, return_intermediate_files, k1=1.5, b = 0.75, alpha=-5, progress=gr.Progress(track_tqdm=True)):
	#bm25.save("saved_df_bm25")
	#bm25 = BM25.load(re.sub(r'\.pkl$', '', file_in.name))

	

	if not in_file:
		out_message ="No input file found. Please load in at least one file."
		print(out_message)
		return out_message, None

	if not corpus:
		out_message = "No data file found. Please load in at least one csv/Excel/Parquet file."
		print(out_message)
		return out_message, None

	if not text_column:
		out_message = "Please enter a column name to search."
		print(out_message)
		return  out_message, None



	file_list = [string.name for string in in_file]

	#print(file_list)

	# Get data file name
	data_file_names = [string for string in file_list if "tokenised" not in string.lower() and "npz" not in string.lower() and "gz" not in string.lower()]

	if not data_file_names:
		return "Please load in at least one csv/Excel/parquet data file.", None

	data_file_name = data_file_names[0]
	data_file_out_name = get_file_path_end_with_ext(data_file_name)
	data_file_name_no_ext = get_file_path_end(data_file_name)

	# Check if there is a search index file already
	#index_file_names = [string for string in file_list if "gz" in string.lower()]

	progress(0.6, desc = "Preparing search index")

	#if index_file_names:
	if search_index:
		#index_file_name = index_file_names[0]

		#print(index_file_name)

		bm25_load = search_index
		

		#index_file_out_name = get_file_path_end_with_ext(index_file_name)
		#index_file_name_no_ext = get_file_path_end(index_file_name)

	else:
		print("Preparing BM25 corpus")

		bm25_load = BM25(corpus, k1=k1, b=b, alpha=alpha)

	global bm25
	bm25 = bm25_load

	if return_intermediate_files == "Yes":
		print("Saving search index file")
		progress(0.8, desc = "Saving search index to file")

		if clean == "Yes":
			bm25_search_file_name = output_folder + data_file_name_no_ext + '_cleaned_search_index.pkl.gz'
		else:
			bm25_search_file_name = output_folder + data_file_name_no_ext + '_search_index.pkl.gz'
			#np.savez_compressed(bm25_search_file_name, bm25)

		with gzip.open(bm25_search_file_name, 'wb') as file:
				pickle.dump(bm25, file)

		print("Search index saved to file")

		message = "Search parameters loaded."

		return message, bm25_search_file_name

	message = "Search parameters loaded."

	print(message)

	return message, None

def convert_bm25_query_to_tokens(free_text_query, clean="No"):
    '''
    Split open text query into tokens and then lemmatise to get the core of the word. Currently 'clean' has no effect.
    '''  

    if clean=="Yes":
        split_query = tokenizer(free_text_query.lower())
        out_query = [token.text for token in split_query]
        #out_query = stem_sentence(out_query)
    else: 
        split_query = tokenizer(free_text_query.lower())
        out_query = [token.text for token in split_query]

    print("Search query out is:", out_query)

    if isinstance(out_query,str):
        print("Converting string")
        out_query = [out_query]

    return out_query

def bm25_search(free_text_query, in_no_search_results, original_data, searched_data, text_column, in_join_file, clean,  in_join_column = "", search_df_join_column = "", progress=gr.Progress(track_tqdm=True)):   

	progress(0, desc = "Conducting keyword search")
	
	# Prepare query
	if (clean == "Yes") | (text_column.endswith("_cleaned")):
		token_query = convert_bm25_query_to_tokens(free_text_query, clean="Yes")
	else:
		token_query = convert_bm25_query_to_tokens(free_text_query, clean="No")

	#print(token_query)

	# Perform search
	print("Searching")

	results_index, results_text, results_scores = bm25.extract_documents_and_scores(token_query, bm25.corpus, n=in_no_search_results) #bm25.corpus #original_data[text_column]
	if not results_index:
		return "No search results found", None

	print("Search complete")

	# Prepare results and export
	joined_texts = [' '.join(inner_list) for inner_list in results_text]
	results_df = pd.DataFrame(data={"index": results_index,
									"search_text": joined_texts,
									"search_score_abs": results_scores})
	results_df['search_score_abs'] = abs(round(results_df['search_score_abs'], 2))

	# Join scores onto searched data
	results_df_out = results_df[['index', 'search_text', 'search_score_abs']].merge(searched_data,left_on="index", right_index=True, how="left", suffixes = ("", "_y")).drop("index_y", axis=1, errors="ignore")

	

	# Join on data from duplicate case notes
	if ("copy_of_case_note_id" in original_data.columns) and ("note_id" in results_df_out.columns):
		if clean == "No":
			print("Clean is no")
			orig_text_column = text_column
		else:
			print("Clean is yes")
			orig_text_column = text_column.replace("_cleaned", "")

		#print(orig_text_column)
		#print(original_data.columns)

		original_data["original_note_id"] = original_data["copy_of_case_note_id"]
		original_data["original_note_id"] = original_data["original_note_id"].combine_first(original_data["note_id"])

		results_df_out = results_df_out.merge(original_data[["original_note_id", "note_id", "copy_of_case_note_id", "person_id"]],left_on="note_id", right_on="original_note_id", how="left", suffixes=("_primary", "")) # .drop(orig_text_column, axis = 1)
		results_df_out.loc[~results_df_out["copy_of_case_note_id"].isnull(), "search_text"] = ""
		results_df_out.loc[~results_df_out["copy_of_case_note_id"].isnull(), text_column] = ""

		#results_df_out = pd.concat([results_df_out, original_data[~original_data["copy_of_case_note_id"].isna()][["copy_of_case_note_id", "person_id"]]])
		# Replace NaN with an empty string
		# results_df_out.fillna('', inplace=True)
		
		
	
	# Join on additional files
	if not in_join_file.empty:
		progress(0.5, desc = "Joining on additional data file")
		join_df = in_join_file
		join_df[in_join_column] = join_df[in_join_column].astype(str).str.replace("\.0$","", regex=True)
		results_df_out[search_df_join_column] = results_df_out[search_df_join_column].astype(str).str.replace("\.0$","", regex=True)

		# Duplicates dropped so as not to expand out dataframe
		join_df = join_df.drop_duplicates(in_join_column)

		results_df_out = results_df_out.merge(join_df,left_on=search_df_join_column, right_on=in_join_column, how="left", suffixes=('','_y'))#.drop(in_join_column, axis=1)

	# Reorder results by score, and whether there is text
	results_df_out = results_df_out.sort_values(['search_score_abs', "search_text"], ascending=False)	

	# Out file
	ensure_output_folder_exists(output_folder)

	query_str_file = ("_").join(token_query)
	results_df_name = output_folder + "keyword_search_result_" + today_rev + "_" +  query_str_file + ".xlsx"

	print("Saving search file output")
	progress(0.7, desc = "Saving search output to file")

	# Highlight found text and save to file
	results_df_out_wb = create_highlighted_excel_wb(results_df_out, free_text_query, "search_text")
	results_df_out_wb.save(results_df_name)
	#results_df_out.to_excel(results_df_name, index= None)
	results_first_text = results_df_out[text_column].iloc[0]

	print("Returning results")

	return results_first_text, results_df_name

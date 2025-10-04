from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import AzureOpenAIEmbeddings
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME')
AZURE_OPENAI_EMBEDDINGS_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_API_KEY')

class facilitesMatcher:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            openai_api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_EMBEDDINGS_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME
        )

    def find_similar_columns(self, df, target_columns):
        df_columns = df.columns.tolist()
        df_embeddings = self.embeddings.embed_documents(df_columns)
        target_embeddings = self.embeddings.embed_documents(target_columns)
        
        similarities = cosine_similarity(target_embeddings, df_embeddings)
        
        column_mapping = {}
        for i, target_col in enumerate(target_columns):
            best_match_index = np.argmax(similarities[i])
            if similarities[i][best_match_index] > 0.7:
                column_mapping[target_col] = df_columns[best_match_index]
        
        return column_mapping

    def map_columns(self, layout_df, comparison_df, selected_columns=None):
        layout_columns = selected_columns or layout_df.columns.tolist()
        comparison_columns = comparison_df.columns.tolist()
        
        layout_embeddings = self.embeddings.embed_documents(layout_columns)
        comparison_embeddings = self.embeddings.embed_documents(comparison_columns)
        
        similarities = cosine_similarity(layout_embeddings, comparison_embeddings)
        
        column_mapping = {}
        mapped_comparison_columns = set()
        for i, layout_col in enumerate(layout_columns):
            if layout_col not in column_mapping:
                best_match_index = np.argmax(similarities[i])
                best_match_col = comparison_columns[best_match_index]
                if similarities[i][best_match_index] > 0.7 and best_match_col not in mapped_comparison_columns:
                    column_mapping[layout_col] = best_match_col
                    mapped_comparison_columns.add(best_match_col)
        
        unmapped_columns = [col for col in comparison_columns if col not in mapped_comparison_columns]
        
        return column_mapping, unmapped_columns

    def find_row_matches(self, reference_df, comparison_df, reference_mapping, comparison_mapping, threshold=0.7):
        if isinstance(reference_mapping, dict):
            reference_columns = list(reference_mapping.values())
        else:
            reference_columns = reference_mapping

        if isinstance(comparison_mapping, dict):
            comparison_columns = list(comparison_mapping.values())
        elif isinstance(comparison_mapping, tuple):
            comparison_columns = list(comparison_mapping[0].values())
        else:
            comparison_columns = comparison_mapping
            
        comparison_columns = [col for col in comparison_columns if col in comparison_df.columns]

        reference_rows = reference_df[reference_columns].astype(str).apply(lambda row: " ".join(row.astype(str)), axis=1).tolist()
        comparison_rows = comparison_df[comparison_columns].astype(str).apply(lambda row: " ".join(row.astype(str)), axis=1).tolist()
        
        reference_embeddings = self.embeddings.embed_documents(reference_rows)
        comparison_embeddings = self.embeddings.embed_documents(comparison_rows)
        
        similarities = cosine_similarity(reference_embeddings, comparison_embeddings)
        
        matches = []
        used_comparison_rows = set()
        for i, row in enumerate(similarities):
            best_match_index = np.argmax(row)
            if row[best_match_index] >= threshold and best_match_index not in used_comparison_rows:
                matches.append((i, best_match_index))
                used_comparison_rows.add(best_match_index)
            else:
                matches.append((i, None))
        
        return matches

    def create_combined_dataframe(self, reference_df, comparison_dfs, all_matches, selected_columns=None, file_column_selections=None):
        all_columns = selected_columns.copy()
        
        for file_name, selected_cols in file_column_selections.items():
            all_columns.extend([f"{file_name}_{col}" for col in selected_cols])
        
        combined_df = pd.DataFrame(columns=all_columns)
        
        for i, reference_row in reference_df.iterrows():
            new_row = {col: reference_row.get(col, None) for col in all_columns if col in reference_df.columns}
            
            for file_name, (comparison_df, unmapped_cols, column_mapping) in comparison_dfs.items():
                matches = all_matches[file_name]
                match = next((m for m in matches if m[0] == i), None)
                if match and match[1] is not None:
                    comparison_row = comparison_df.iloc[match[1]]
                    for layout_col, comparison_col in column_mapping.items():
                        if layout_col in all_columns and (layout_col not in new_row or pd.isna(new_row[layout_col])):
                            new_row[layout_col] = comparison_row.get(comparison_col, None)
                    for col in file_column_selections[file_name]:
                        new_row[f"{file_name}_{col}"] = comparison_row.get(col, None)
                else:
                    for col in file_column_selections[file_name]:
                        new_row[f"{file_name}_{col}"] = None
            
            combined_df = pd.concat([combined_df, pd.DataFrame([new_row])], ignore_index=True)
        
        return combined_df

def fix_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df
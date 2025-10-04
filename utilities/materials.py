import json
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from openpyxl.styles import Alignment
from langchain_openai import AzureOpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_type = os.getenv("AZURE_OPENAI_TYPE")
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME')


class OrderedEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, OrderedDict):
                return {k: self.default(v) for k, v in obj.items()}
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
        
def read_excel_preserve_format(file):
    # Read the Excel file
    df = pd.read_excel(file, dtype=object)
    
    def preserve_format(value):
        if pd.isna(value):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        return value

    # Apply the preserve_format function to all cells
    for column in df.columns:
        df[column] = df[column].apply(preserve_format)
    
    return df

class ExcelMatcher:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=openai.api_version,
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
        if selected_columns:
            all_columns = selected_columns.copy()
        else:
            target_columns = ["Description", "Qty", "UOM", "Manufacturer"]
            all_columns = [col for col in reference_df.columns if col in target_columns]
       
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
   
    @staticmethod
    def format_excel(workbook):
        ws = workbook.active
       
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')
       
        for column in ws.columns:
            max_length = 0
            column = [cell for cell in column if cell.value is not None]
            if column:
                max_length = max(len(str(cell.value)) for cell in column)
            adjusted_width = (max_length + 2) * 1.2
            ws.column_dimensions[column[0].column_letter].width = adjusted_width


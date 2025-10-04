from langchain.globals import set_debug
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
import tempfile
import xlrd
import pandas as pd
import os
import time
from flask import jsonify
import json
import re
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import logging
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import openai
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Dict, Any, Optional, Union, Tuple, List
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part
from langchain.schema import Document

load_dotenv()

# Set up basic configuration for the logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger()

# Add Azure AI Document Intelligence API credentials
DOCAI_ENDPOINT = os.getenv('AZURE_DOCUMENTAI_ENDPOINT')
DOCAI_KEY = os.getenv('AZURE_DOCUMENTAI_KEY')

openai.api_type = os.getenv("AZURE_OPENAI_TYPE")
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
LLM_MODEL = os.getenv('AZURE_OPENAI_GPT4_DEPLOYMENT_NAME')
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME')


llm = AzureChatOpenAI(
    azure_deployment=LLM_MODEL,
    openai_api_key=openai.api_key,
    openai_api_version=openai.api_version,
    azure_endpoint=openai.api_base
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=openai.api_version,
)

class ExcelAnalyzer:
    def __init__(self, credentials_path, prompt_template):
        """
        Initialize the Excel analyzer with Vertex AI (Gemini) integration
        
        Args:
            credentials_path (str): Path to GCP service account credentials
            prompt_template (str): System prompt template for analysis
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Get project ID from credentials
        with open(credentials_path, 'r') as f:
            project_id = json.load(f)['project_id']
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location="us-central1"
        )
        
        # Initialize the model
        self.model = GenerativeModel("gemini-1.5-flash-002")
        self.prompt_template = prompt_template
        self.filled_columns = []
        self.not_filled_columns = []
        self.extracted_data = []
        self.df = None

    def create_table_image(self, file_path):
        """Create a table visualization from Excel file and return it as a PIL Image"""
        # Load the Excel file
        self.df = pd.read_excel(file_path)
        
        # Replace NaN values with blank spaces
        self.df.fillna('', inplace=True)
        
        # Remove unnamed columns
        self.df.columns = [col if 'Unnamed' not in col else '' for col in self.df.columns]
        
        # Select the first 20 rows
        df_first_20 = self.df.head(20)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df_first_20.values, 
                        colLabels=df_first_20.columns, 
                        cellLoc='center', 
                        loc='center')
        
        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert buffer to PIL Image
        buf.seek(0)
        return Image.open(buf)

    def analyze_excel_file(self, file_path, prompt, json_val=False):
        """
        Process Excel file and get Gemini Vision analysis
        
        Args:
            file_path (str): Path to the Excel file
            prompt (str): Prompt for analysis
            json_val (bool): Whether to return raw JSON response
            
        Returns:
            dict: Analysis results or error message
        """
        try:
            # Create table image
            image = self.create_table_image(file_path)
            
            # Convert image to bytes for Gemini
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create image part for Gemini
            image_part = Part.from_data(data=img_byte_arr, mime_type="image/png")
            
            # Generate content with Gemini
            response = self.model.generate_content(
                [self.prompt_template, prompt, image_part],
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0,
                    "top_p": 1,
                }
            )
            
            # Get response text
            llm_response = response.text
            
            print("llm_response", llm_response)
            
            # Extract JSON from response
            cleaned_string = re.search(r'```json(.*?)```', llm_response, re.DOTALL)
            if cleaned_string:
                cleaned_string = cleaned_string.group(1).strip()
            else:
                cleaned_string = llm_response.strip()
                
            if json_val:
                return cleaned_string
            
            try:
                column_status = json.loads(cleaned_string)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response from LLM"}
            
            # Store filled columns
            self.filled_columns = [col for col, status in column_status.items() if status.lower() == "filled"]
            self.not_filled_columns = [col for col, status in column_status.items() if status.lower() == "not filled"]
            
            if len(self.filled_columns) == 0:
                print("No filled columns found in the Excel file.")
                return {"error": "No columns found in the Excel file"}
            
            elif len(self.not_filled_columns) == 0:
                print("No to be filled columns found in the Excel file.")
                # return {"error": 'No to be filled columns found in the Excel file.'}
                
            return {
                "filled_columns": self.filled_columns,
                "not_filled_columns": self.not_filled_columns
            }
            
        except Exception as e:
            return {"error": str(e)}

    def get_filled_columns_dataframe(self):
        """
        Return a list of concatenated values from filled columns, handling misaligned headers.
        Searches for column headers that may appear in different rows and reconstructs the data accordingly.
        """
        if not hasattr(self, 'df') or not self.filled_columns:
            return []

        result_list = []
        header_positions = {}  # Store {column_name: (row_idx, col_idx)}
        search_range = 20  # Number of initial rows to search for headers

        # First pass: locate all target headers
        for target_header in self.filled_columns:
            header_found = False
            
            # Check if header is in column names
            if target_header in self.df.columns:
                header_positions[target_header] = (-1, self.df.columns.get_loc(target_header))
                header_found = True
                continue

            # Search through initial rows
            for row_idx in range(min(search_range, len(self.df))):
                for col_idx, value in enumerate(self.df.iloc[row_idx]):
                    if pd.notna(value) and str(value).strip() == target_header:
                        header_positions[target_header] = (row_idx, col_idx)
                        header_found = True
                        break
                if header_found:
                    break

            if not header_found:
                logger.warning(f"Header '{target_header}' not found in first {search_range} rows")
                return []

        # Determine the effective data start row (maximum header row index + 1)
        data_start_idx = max(row_idx for row_idx, _ in header_positions.values() if row_idx >= 0) + 1 if any(row_idx >= 0 for row_idx, _ in header_positions.values()) else 0

        # Create a new DataFrame with properly aligned columns
        new_data = {}
        for header, (row_idx, col_idx) in header_positions.items():
            if row_idx == -1:  # Header was in column names
                new_data[header] = self.df[header].iloc[data_start_idx:].tolist()
            else:
                new_data[header] = self.df.iloc[data_start_idx:, col_idx].tolist()

        result_df = pd.DataFrame(new_data)

        # Clean up the DataFrame
        result_df = result_df.dropna(how='all')
        result_df.fillna('', inplace=True)
        result_df = result_df.reset_index(drop=True)

        # Add validation to ensure data alignment
        if len(result_df.columns) != len(self.filled_columns):
            logger.warning("Mismatch between expected and found columns")
            return []

        # Convert DataFrame rows to list of concatenated strings with additional validation
        for _, row in result_df.iterrows():
            valid_entries = []
            for col, val in row.items():
                val_str = str(val).strip()
                if val_str and not val_str.lower() in ['nan', 'none', 'null']:
                    valid_entries.append(f"{col}: {val_str}")
            
            if valid_entries:  # Only add rows with valid data
                result_list.append(", ".join(valid_entries))

        return result_list

    def validate_header_alignment(self, header_positions):
        """
        Validate that found headers make logical sense in their positions.
        Returns True if the header positions appear valid, False otherwise.
        """
        row_positions = sorted(set(row for row, _ in header_positions.values() if row >= 0))
        
        # Check if headers are too far apart vertically
        if len(row_positions) > 0 and row_positions[-1] - row_positions[0] > 3:
            logger.warning("Headers are spread across too many rows")
            return False
        
        # Check for column overlap
        col_positions = [col for _, col in header_positions.values()]
        if len(col_positions) != len(set(col_positions)):
            logger.warning("Multiple headers found in the same column")
            return False
            
        return True

    def extract_table_data(self, header_positions, data_start_idx):
        """
        Extract data from the table based on validated header positions.
        Handles special cases and data type conversions.
        """
        data_dict = {}
        
        for header, (row_idx, col_idx) in header_positions.items():
            column_data = []
            
            if row_idx == -1:  # Header in column names
                column_data = self.df[header].iloc[data_start_idx:].tolist()
            else:
                column_data = self.df.iloc[data_start_idx:, col_idx].tolist()
                
            # Clean and validate the data
            cleaned_data = []
            for value in column_data:
                if pd.isna(value):
                    cleaned_data.append('')
                else:
                    # Handle different data types appropriately
                    if isinstance(value, (int, float)):
                        cleaned_data.append(str(value))
                    else:
                        cleaned_data.append(str(value).strip())
                        
            data_dict[header] = cleaned_data
            
        return pd.DataFrame(data_dict)


def read_xls_to_prompts(file):
    """
    Read an Excel file and extract prompts based on filled columns analysis.

    Args:
        file: FileStorage object from Flask request

    Returns:
        list: [prompts, non_filled, error_message or None]
    """
    # Only accept .xlsx files
    if not file.filename.endswith('.xlsx'):
        logging.error("File format not supported. Only .xlsx files are accepted.")
        return [None, None, "Only .xlsx files are supported."]

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    try:
        # Initialize ExcelAnalyzer
        prompt_template = """You are an assistant that analyzes Excel spreadsheets and identifies filled and empty columns."""
        analyzer = ExcelAnalyzer("KEY.json",prompt_template)
        
        analysis_prompt = """Given an image of a table:
 
1. Identify all column headers in the table.
 
2. For each column below its header:
   - STRICTLY check for visible alphanumeric content (text, numbers).
   - Ignore any empty spaces, borders, or formatting.
   - Ignore column headers themselves when checking if a column is filled.
   - Only consider the rows below the header for determining if a column is filled.
 
3. Mark a column as 'not filled' if and ONLY if:
   - You cannot see any actual text or numbers in ANY cells below the header.
   - Empty cells, spaces, or just formatting don't count as filled.
   - If text or numeric values are present above the header consider as 'not filled'
   - If you're unsure if there's content, mark it as 'not filled'.
   - Ensure that any non-visible or minimal content is not considered as filled.
   - Even if there is minimal or unclear content, mark it as 'not filled'.
 
4. Mark a column as 'filled' if and ONLY if:
   - You can clearly see at least one cell with actual text or numbers below the header.
   - The content must be visibly readable, not just formatting.
 
 
5. Return a JSON object where:
   - Keys are the exact column header names.
   - Values are ONLY either "filled" or "not filled".
   - Include all columns, even if they appear empty.
 
Example format:
```json
{
    "Column Header 1": "filled",
    "Column Header 2": "not filled"
}
'''
"""
        
        try:
            result = analyzer.analyze_excel_file(temp_path, analysis_prompt, json_val=False)
        except Exception as e:
            logging.error(f"Error analyzing Excel file: {str(e)}")
            return [None, None, f"Error analyzing Excel file: {str(e)}"]

        # Check if result is a dictionary with 'error'
        if isinstance(result, dict) and 'error' in result:
            logging.error(f"Error in analysis result: {result['error']}")
            return [None, None, result['error']]

        # Get the filled columns data
        try:
            prompts = analyzer.get_filled_columns_dataframe()
        except Exception as e:
            logging.error(f"Error extracting filled columns: {str(e)}")
            return [None, None, f"Error extracting filled columns: {str(e)}"]
        
        try:
            not_filled = analyzer.not_filled_columns
        except Exception as e:
            logging.error(f"Error extracting not filled columns: {str(e)}")
            return [None, None, f"Error extracting not filled columns: {str(e)}"]
        
        # Handle case with no filled columns
        if not prompts:
            logging.warning("No filled columns found in the Excel file.")
            return [None, None, "No filled columns found in the Excel file."]

        logging.info(f"Extracted {len(prompts)} prompts from the Excel file.")
        return [prompts, not_filled, None]

    except Exception as e:
        logging.error("Provide proper excel format")
        return [None, None, f"Error processing file: {str(e)}"]

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logging.error(f"Error removing temporary file: {str(e)}")
            
def process_excel_with_pandas(file_path: str) -> List[Document]:
    """
    Process Excel file using pandas and convert it to LangChain Document objects.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        List of LangChain Document objects containing the processed content
    """
    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(file_path)
        documents = []
        
        for sheet_name in excel_file.sheet_names:
            # Read each sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Convert any non-string columns to string
            df = df.astype(str)
            
            # Create a text representation of the sheet
            content = []
            
            # Add sheet name as header
            content.append(f"Sheet: {sheet_name}")
            
            # Add column headers
            content.append("Headers: " + " | ".join(df.columns.tolist()))
            
            # Add row data
            for idx, row in df.iterrows():
                row_content = " | ".join(row.values)
                content.append(f"Row {idx + 1}: {row_content}")
            
            # Join all content with newlines
            text_content = "\n".join(content)
            
            # Create a LangChain Document object
            document = Document(
                page_content=text_content,
                metadata={
                    "source": file_path,
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            )
            documents.append(document)
            
        return documents
        
    except Exception as e:
        raise ValueError(f"Error processing Excel file: {str(e)}")


def process_single_file(file):
    """
    Process a single file (PDF, DOCX, or XLSX) and return a tool or None
    
    Args:
        file: File object to process
        
    Returns:
        Tool: Langchain Tool object for document QA or None if processing fails
    """
    temp_file = None
    try:
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Create temporary file with appropriate extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        file.save(temp_file.name)
        temp_file.close()
        
        # Select appropriate loader based on file type
        if file_extension == '.pdf' or file_extension == '.docx':
            doc_loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=DOCAI_ENDPOINT,
                api_key=DOCAI_KEY,
                file_path=temp_file.name,
                api_model="prebuilt-layout"
            )
            documents = doc_loader.load()

        elif file_extension == '.xlsx':
            try:
                documents = process_excel_with_pandas(temp_file.name)
            except ValueError as e:
                raise ValueError(f"Error processing Excel file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load and process documents
        print("document :",documents)
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        docs = text_splitter.split_documents(documents)
        
        if docs:
            # Create retriever from documents
            retriever = FAISS.from_documents(docs, embeddings).as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 20}
            )
            
            # Create and return tool
            return Tool(
                name=file.filename,
                description=f"Extracts information from {file.filename}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            )
        return None
        
    except Exception as e:
        print(f"Error processing {file.filename}: {str(e)}")
        return None
        
    finally:
        if temp_file and temp_file.name:
            safe_file_remove(temp_file.name)


# Helper function to safely remove temporary files
def safe_file_remove(filepath):
    """
    Safely remove a file if it exists
    
    Args:
        filepath: Path to file to remove
    """
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
    except Exception as e:
        print(f"Error removing temporary file {filepath}: {str(e)}")

def extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON from text, with advanced repair for truncated JSON.
    
    Args:
        text (str): Input text containing JSON data, potentially truncated
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary or None if parsing fails
    """
    def clean_text(t: str) -> str:
        """Clean and prepare text for parsing."""
        # Remove 'result' wrapper if present
        if t.startswith('{"result":'):
            t = t[len('{"result":'):]
            
        # Remove markdown code block syntax
        t = re.sub(r'```\s*(?:json)?\s*', '', t)
        
        # Clean up escaped newlines and quotes
        t = t.replace('\\n', '\n').replace('\\"', '"')
        return t.strip()

    def analyze_json_structure(t: str) -> Tuple[int, int, int, int]:
        """
        Analyze JSON structure to count unclosed brackets and braces.
        Returns counts of open and closed braces and brackets.
        """
        open_braces = t.count('{')
        close_braces = t.count('}')
        open_brackets = t.count('[')
        close_brackets = t.count(']')
        return open_braces, close_braces, open_brackets, close_brackets

    def repair_truncated_json(t: str) -> str:
        """
        Advanced repair for truncated JSON with smart structure completion.
        """
        # Handle empty or severely malformed input
        if not t or not t.strip():
            return "{}"
        
        # Remove any trailing comma followed by whitespace at the end
        t = re.sub(r',\s*$', '', t)
        
        # Analyze current structure
        open_braces, close_braces, open_brackets, close_brackets = analyze_json_structure(t)
        
        # If we have a truncated string value, complete it
        if t.count('"') % 2 == 1:
            t = t + '"'
            
        # Complete any object values that were cut off
        if re.search(r':\s*$', t):
            t = t + '""'
            
        # Add missing closing braces and brackets
        missing_braces = open_braces - close_braces
        missing_brackets = open_brackets - close_brackets
        
        # First close any open brackets (arrays)
        if missing_brackets > 0:
            t = t + ']' * missing_brackets
            
        # Then close any open braces (objects)
        if missing_braces > 0:
            t = t + '}' * missing_braces
            
        return t

    def convert_numeric_keys(data: Union[Dict, Any]) -> Union[Dict, Any]:
        """Convert string numeric keys to integers and sort them."""
        if not isinstance(data, dict):
            return data
            
        converted_dict = {}
        for key, value in data.items():
            # Convert string numeric keys to integers
            try:
                if isinstance(key, str) and key.isdigit():
                    new_key = int(key)
                else:
                    new_key = key
            except (ValueError, TypeError):
                new_key = key
                
            # Recursively convert nested structures
            if isinstance(value, dict):
                converted_dict[new_key] = convert_numeric_keys(value)
            elif isinstance(value, list):
                converted_dict[new_key] = [convert_numeric_keys(item) if isinstance(item, dict) else item for item in value]
            else:
                converted_dict[new_key] = value
                
        return converted_dict

    def normalize_values(data: Dict) -> Dict:
        """Normalize and clean up values in the dictionary."""
        if not isinstance(data, dict):
            return data
            
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = normalize_values(value)
            elif isinstance(value, str):
                # Clean up currency values
                if value.startswith('$'):
                    value = value.replace('$', '').replace(',', '')
                result[key] = value.strip()
            else:
                result[key] = value
        return result

    try:
        # Initial cleaning
        cleaned_text = clean_text(text)
        
        # First try: Direct parse
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Second try: Repair and parse
            repaired_text = repair_truncated_json(cleaned_text)
            try:
                parsed_json = json.loads(repaired_text)
            except json.JSONDecodeError as e:
                # If we still can't parse, try to find the last valid object
                logging.warning(f"Could not parse repaired JSON, attempting to find last valid object: {str(e)}")
                # Find last complete object by looking for pattern: "digits": { ... }
                pattern = r'("127"\s*:\s*{[^}]*})'
                match = re.search(pattern, cleaned_text)
                if match:
                    # Extract up to this point and repair
                    truncated_text = cleaned_text[:match.end()]
                    final_repaired = repair_truncated_json(truncated_text + "}")
                    parsed_json = json.loads(final_repaired)
                else:
                    raise
        
        # Handle nested JSON in 'result' key
        if isinstance(parsed_json, dict) and 'result' in parsed_json:
            try:
                if isinstance(parsed_json['result'], str):
                    inner_json = json.loads(parsed_json['result'])
                    parsed_json = inner_json
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Convert string numeric keys to integers
        converted_json = convert_numeric_keys(parsed_json)
        
        # Normalize values
        normalized_json = normalize_values(converted_json)
        
        # Sort dictionary with numeric keys while preserving non-numeric keys
        if isinstance(normalized_json, dict) and 'prompts' in normalized_json:
            prompts_dict = normalized_json['prompts']
            sorted_prompts = dict(sorted(prompts_dict.items(), key=lambda x: (isinstance(x[0], str), x[0])))
            normalized_json['prompts'] = sorted_prompts
        
        return normalized_json
        
    except Exception as e:
        logging.error(f"Error processing JSON: {str(e)}")
        return None



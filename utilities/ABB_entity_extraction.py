import re
import PyPDF2
from azure.storage.blob import BlobServiceClient
import io
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from utilities.Document_Processing import DocumentIntelligenceLoader
from utilities.Document_Processing import BlobStorageProcessor
load_dotenv()

AZURE_BLOB_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_BLOB_STORAGE_CONNECTION_STRING')
BLOB_URL=os.environ.get('BLOB_URL')
documentintelligenceloader=DocumentIntelligenceLoader()
blobstorageprocessor=BlobStorageProcessor()

def extract_entities_from_blob(container_name,blob_name):

    # blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_STORAGE_CONNECTION_STRING)

    # blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    # # # Download the blob content
    # download_stream = blob_client.download_blob()
    
    # # # Create an in-memory bytes buffer
    # pdf_content = io.BytesIO(download_stream.readall())
    if blob_name.endswith('.pdf'):
        blob_url=blobstorageprocessor.get_blob_sas_url(container_name,blob_name)
        result = documentintelligenceloader.analyze_document_word(blob_url)
        lines = [line.content for page in result.pages for line in page.lines]
    elif blob_name.endswith('.docx'):
        blob_url=blobstorageprocessor.get_blob_sas_url(container_name,blob_name)
        result = documentintelligenceloader.analyze_document_word(blob_url)
        lines = [paragraph.content for paragraph in result.paragraphs]
    # Create a PDF reader object
    # reader = PyPDF2.PdfReader(pdf_content)
    
    # Extract text from all pages
    # extracted_text = ""
    # for page in reader.pages:
    #     extracted_text += page.extract_text()
    
    # Define a regular expression pattern for the heading format
    heading_pattern = r"\d+\.?\s+[A-Z' ]+(?:, [A-Z ]+)*(?:\s*\n\s*[A-Z ]+)*"
    
    # Initialize dictionary to store headings and data
    headings_and_data = {}
    
    # Initialize variables to track current heading and data
    current_heading = None
    current_data = []
    
    # Split the extracted text into lines
    # lines = extracted_text.split('\n')
    
    # Process each line to extract headings and data
    for line in lines:
        line = line.strip()
        if re.match(heading_pattern, line):
            # Found a new heading
            if current_heading:
                # Store previous heading and data in the dictionary
                headings_and_data[current_heading] = '\n'.join(current_data)
                current_data = []
            current_heading = line
        elif current_heading:
            # Add line to data under current heading
            current_data.append(line)
    
    # Store the last heading and its data
    if current_heading:
        headings_and_data[current_heading] = '\n'.join(current_data)
    
    return headings_and_data



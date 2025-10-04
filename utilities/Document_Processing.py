import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobSasPermissions, generate_blob_sas
# from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat, AnalyzeResult

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_BLOB_STORAGE_CONNECTION_STRING')
AZURE_BLOB_STORAGE_ACCOUNT_NAME=os.environ.get("AZURE_BLOB_STORAGE_ACCOUNT_NAME")
AZURE_BLOB_STORAGE_ACCOUNT_KEY=os.environ.get("AZURE_BLOB_STORAGE_ACCOUNT_KEY")


DOCAI_ENDPOINT = os.environ.get('AZURE_DOCUMENTAI_ENDPOINT')
DOCAI_KEY = os.environ.get('AZURE_DOCUMENTAI_KEY')

class BlobStorageProcessor:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    def upload_blob(self,container_name,blob_name, data):
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data,overwrite=True)

    def get_blob_sas_url(self, container_name,blob_name):
        sas_token = generate_blob_sas(
            account_name=AZURE_BLOB_STORAGE_ACCOUNT_NAME,
            container_name=container_name,
            blob_name=blob_name,
            account_key=AZURE_BLOB_STORAGE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        return "https://"+AZURE_BLOB_STORAGE_ACCOUNT_NAME+".blob.core.windows.net/"+container_name+"/"+blob_name+"?"+sas_token
    
    def download_blob(self,container_name,blob_name):
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.blob_client = self.container_client.get_blob_client(blob_name)
        return self.blob_client.download_blob()
    
class DocumentIntelligenceLoader:
    def __init__(self):
        self.document_intelligence_client = DocumentIntelligenceClient(endpoint=DOCAI_ENDPOINT, credential=AzureKeyCredential(DOCAI_KEY))

    def analyze_document_pdf(self, blob_sas_url):
        poller = self.document_intelligence_client.begin_analyze_document("prebuilt-layout",AnalyzeDocumentRequest(url_source=blob_sas_url),output_content_format=ContentFormat.MARKDOWN,)
        result = poller.result()
        return result
    
    def analyze_document_word(self, blob_sas_url):
        poller = self.document_intelligence_client.begin_analyze_document("prebuilt-read",AnalyzeDocumentRequest(url_source=blob_sas_url))
        result = poller.result()
        return result
import os
import os
import tempfile
import shutil
import time
import logging
import json
import re
from typing import List, Dict, Any, Optional
from git import Repo
from  pinecone import Pinecone
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import httpx
from utils.config import get_app_config, get_pinecone_config
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from urllib.parse import urlparse


# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BotService:
    def __init__(self, fastapi_url: str):
        self.fastapi_url = "https://api.creditchek.africa"
        self.forced_language = None
        self.current_frameworks = []
        
        # Sample documents for document retrieval, including all Kenya credit report endpoints
        self.sample_docs = [
            Document(page_content="RecovaPRO SDK: Initialize with API key for CreditChek API access.", metadata={"source": "sdk_docs"}),
            Document(page_content="Kenya Consumer Variable Report: GET /v1/credit/ke/individual-variableReport with lastName, firstName, nationalID.", metadata={"source": "api_docs"}),
            Document(page_content="Kenya Basic Report: GET /v1/credit/ke/individual with lastName, firstName, nationalID.", metadata={"source": "api_docs"}),
            Document(page_content="Kenya Full Report: GET /v1/credit/ke/individual-premium with lastName, firstName, nationalID.", metadata={"source": "api_docs"}),
            Document(page_content="Kenya Consumer Variable Report with Score: GET /v1/credit/ke/individual-variableScoreReport with lastName, firstName, nationalID.", metadata={"source": "api_docs"}),
            Document(page_content="Kenya Mobile Loan History Report: GET /v1/credit/ke/individual-mobileReport-history with lastName, firstName, nationalID.", metadata={"source": "api_docs"}),
            Document(page_content="Kenya Mobile Loan Report: GET /v1/credit/ke/individual-mobileReport with lastName, firstName, nationalID.", metadata={"source": "api_docs"}),
            Document(page_content="Kenya Individual Verification: POST /v1/identity/ke/verifyIndividual with id_type, identifier, first_name, last_name.", metadata={"source": "api_docs"}),
            Document(page_content="Nigeria BVN Verification: POST /api/v1/identity/verifyData with type=bvn, response_type=bvn.", metadata={"source": "api_docs"}),
        ]
        
        try:
            self.app_config = get_app_config()
            if not isinstance(self.app_config, dict):
                raise ValueError("get_app_config must be a dictionary")
                
            self.app_config["supported_frameworks"] = {
                "Python": ["Flask", "FastAPI", "Django"],
                "JavaScript": ["Express", "Koa", "NodeJS"],
                "PHP": ["Laravel", "Slim"],
                "Go": ["net/http", "Gin", "Echo"],
                "Java": ["Spring", "Jakarta EE"],
                "C#": [".NET", "ASP.NET Core"],
                "Ruby": ["Rails", "Sinatra"],
                "Rust": ["Actix", "Rocket"],
                "Swift": ["Vapor", "Perfect"],
                "Kotlin": ["Ktor", "Spring Boot"],
                "Other": ["Standard Library"]
            }
            
            if "supported_languages" not in self.app_config:
                self.app_config["supported_languages"] = list(self.app_config["supported_frameworks"].keys())
                
            if "preconfigured_prompts" not in self.app_config:
                self.app_config["preconfigured_prompts"] = [
                    "How do I set up webhooks for CreditChek transaction updates?",
                    "How do I get started with the CreditChek APIs?",
                    "What are the different service endpoints available in the CreditChek APIs?",
                    "What are the required headers and parameters for each endpoint?",
                    "What are the available response formats for each endpoint?",
                    "How do I authenticate and generate an access token for the CreditChek APIs?",
                    "Are there any usage limits or rate limits for the CreditChek APIs?",
                    "How do I handle errors and interpret error codes in the CreditChek APIs?",
                    "How can I test the CreditChek APIs in a sandbox environment before going live?",
                    "Get consumer variable report for Kenya",
                    "Get basic report for Kenya",
                    "Get full report for Kenya",
                    "Get consumer variable report with score for Kenya",
                    "Get mobile loan history report for Kenya"
                ]
        except Exception as e:
            logger.error(f"Failed to load app_config: {e}")
            raise ValueError("Invalid app configuration")
            
        try:
            self.pinecone_config = get_pinecone_config()
            if not isinstance(self.pinecone_config, dict):
                raise ValueError("get_pinecone_config must return a dictionary")
            if not self.pinecone_config.get("PINECONE_INDEX"):
                self.pinecone_config["PINECONE_INDEX"] = "creditchek-docs"
        except Exception as e:
            logger.error(f"Failed to load pinecone_config: {e}")
            raise
        
        self.pinecone_client = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY") or self.pinecone_config.get("api_key"),
        environment=os.getenv("PINECONE_ENVIRONMENT") or self.pinecone_config.get("environment")
    )
        # Get index handle
        self.index = self.pinecone_client.Index(self.pinecone_config.get("PINECONE_INDEX", "creditchek-docs"))
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.agent_prompt = PromptTemplate.from_template(
            """You are Kasi, an AI assistant for CreditChek API integration.
            Plan a response for the query in {language} using {framework}.
            Return a JSON object with:
            - plan: { steps: [{ tool: string, params: object, description: string }] }
            Query: {query}
            Chat History: {chat_history}
            """
        )
        self.init_document_store()

    def set_language_preference(self, language: str):
        self.forced_language = language
        logger.info(f"Language preference set to: {language}")
        if language in self.app_config["supported_frameworks"]:
            self.current_frameworks = self.app_config["supported_frameworks"][language]
        else:
            self.current_frameworks = ["Standard Library"]
        logger.debug(f"Available frameworks: {self.current_frameworks}")

    def create_index(self):
        index_name = self.pinecone_config.get("PINECONE_INDEX", "creditchek-docs")
        try:
            existing_indexes = self.pinecone_instance.list_indexes()
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pinecone_instance.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine"
                )
                time.sleep(5)  # Wait for index creation
            else:
                logger.info(f"Pinecone index {index_name} already exists.")
            return self.pinecone_instance.Index(index_name)
        except Exception as e:
            logger.error(f"Failed to create/connect to index: {e}")
            raise

    def init_document_store(self):
        logger.info("Initializing document store with sample documents")
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info(f"Pinecone index contains {total_vectors} vectors")
            if total_vectors == 0:
                logger.info("Index is empty, uploading sample documents")
                self._upload_documents_in_batches(self.sample_docs, batch_size=5)
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            raise

    def is_index_populated(self) -> bool:
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0) > 0
        except Exception as e:
            logger.error(f"Failed to check index status: {e}")
            return False

    def process_github_repo(self, repo_url: str, branch: str = "master", force_reprocess: bool = False):
        if not force_reprocess and self.is_index_populated():
            logger.info(f"Index {self.pinecone_config['PINECONE_INDEX']} already populated, skipping processing")
            return
        temp_dir = tempfile.mkdtemp()
        try:
            logger.info(f"Cloning repository {repo_url} (branch: {branch}) to {temp_dir}")
            repo = Repo.clone_from(repo_url, temp_dir, branch=branch)
            docs_path = os.path.join(temp_dir, "docs")
            if not os.path.exists(docs_path):
                logger.warning(f"No 'docs' directory found in repository {repo_url}")
                return
            documents = []
            for root, _, files in os.walk(docs_path):
                for file in files:
                    if file.endswith((".md", ".mdx")):
                        file_path = os.path.join(root, file)
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
            logger.info(f"Loaded {len(documents)} documents from repository")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            self._upload_documents_in_batches(split_docs, batch_size=50)
        except Exception as e:
            logger.error(f"Failed to process repository {repo_url}: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _upload_documents_in_batches(self, documents: List[Document], batch_size: int):
        start_time = time.time()
        logger.info(f"Uploading {len(documents)} documents in batches of {batch_size}")
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, batch in enumerate(batches):
                try:
                    start_batch = time.time()
                    vectors = self._create_vectors(batch, i)
                    self.index.upsert(vectors=vectors, namespace=self.pinecone_config.get("namespace", None))
                    logger.info(f"Batch {i + 1}/{len(batches)} uploaded in {time.time() - start_batch:.2f} seconds")
                except Exception as e:
                    logger.error(f"Failed to upload batch {i + 1}: {e}")
                    raise
        logger.info(f"Indexed {len(documents)} document chunks in {time.time() - start_time:.2f} seconds")

    def _create_vectors(self, batch: List[Document], batch_index: int) -> List[Dict[str, Any]]:
        try:
            embeddings = self.embeddings.embed_documents([doc.page_content for doc in batch])
            vectors = [
                {
                    "id": f"doc_{batch_index * 1000 + i}",
                    "values": embedding,
                    "metadata": {
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content[:500]
                    }
                }
                for i, (doc, embedding) in enumerate(zip(batch, embeddings))
            ]
            return vectors
        except Exception as e:
            logger.error(f"Failed to create vectors for batch: {e}")
            raise

    def _document_retrieval_tool(self, query: str) -> List[Document]:
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            if total_vectors == 0:
                logger.warning("Index is empty, using sample documents")
                return [doc for doc in self.sample_docs if query.lower() in doc.page_content.lower()][:3]
            query_embedding = self.embeddings.embed_query(query)
            query_results = self.index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                namespace=self.pinecone_config.get("namespace", None)
            )
            return [
                Document(
                    page_content=match.get("metadata", {}).get("content", "No content available"),
                    metadata=match.get("metadata", {})
                )
                for match in query_results.get("matches", [])
            ]
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return [doc for doc in self.sample_docs if query.lower() in doc.page_content.lower()][:3]

    def _creditchek_api_tool(self, endpoint: str, params: Dict[str, Any] = None, country: str = "nigeria") -> Dict[str, Any]:
        if not params:
            params = {}
        
        api_key = os.getenv("CREDITCHEK_API_KEY") or self.pinecone_config.get("api_key")
        if not api_key:
            logger.error("CREDITCHEK_API_KEY environment variable not set")
            return {"error": "CREDITCHEK_API_KEY environment variable not set"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        country = country.lower()
        if country not in ["nigeria", "kenya"]:
            logger.error(f"Unsupported country: {country}")
            return {"error": f"Unsupported country: {country}"}

        try:
            with httpx.Client() as client:
                if endpoint == "/api/v1/identity/verifyData":
                    required_params = ["type"]
                    if not all(param in params for param in required_params):
                        missing = [param for param in required_params if param not in params]
                        logger.error(f"Missing required parameters: {missing}")
                        return {"error": f"Missing required parameters: {missing}"}
                    if params["type"] == "nin" and "nin" not in params:
                        return {"error": "NIN parameter required for NIN verification"}
                    if params["type"] == "bvn" and "bvn" not in params:
                        return {"error": "BVN parameter required for BVN verification"}
                    if params["type"] == "cac" and "cac_number" not in params:
                        return {"error": "CAC number required for CAC verification"}
                    if params["type"] == "driver_license" and "driver_license" not in params:
                        return {"error": "Driver’s License number required for Driver’s License verification"}
                    if params["type"] == "bank_account" and not all(k in params for k in ["account_number", "bank_code"]):
                        return {"error": "Account number and bank code required for Bank Account verification"}
                    response = client.post(
                        f"{self.fastapi_url}{endpoint}",
                        json=params,
                        headers=headers,
                        timeout=10.0
                    )
                elif endpoint == "/v1/income/api/process-income" and country == "nigeria":
                    required_params = ["bvn", "accountName", "accountNumber", "accountType", "bankName", "bankCode", "statement"]
                    missing = [p for p in required_params if p not in params]
                    if missing:
                        logger.warning(f"Simulation mode: missing params {missing}")
                        return {
                            "info": (
                                f"Endpoint `/v1/income/api/process-income` requires the following fields: "
                                f"{', '.join(required_params)}. Please provide all to run it live."
                            )
                        }

                    # Only proceed if real file exists
                    if not os.path.exists(params["statement"]):
                        logger.warning(f"Simulation mode: Statement file missing at {params['statement']}")
                        return {
                            "info": (
                                "To execute this API, you must upload a valid PDF bank statement at "
                                f"`{params['statement']}`. Until then, here’s how the API works..."
                            )
                        }
                elif endpoint in ("/v1/credit/credit-registry", "/v1/credit/crc", "/v1/credit/first-central", "/v1/credit/advanced", "/v1/credit/premium") and country == "nigeria":
                    if "bvn" not in params:
                        logger.error("BVN parameter required for individual credit insights")
                        return {"error": "BVN parameter required for individual credit insights"}
                    response = client.get(
                        f"{self.fastapi_url}{endpoint}",
                        params={"bvn": params["bvn"]},
                        headers=headers,
                        timeout=10.0
                    )
                elif endpoint in ("/v1/credit/sme/crc", "/v1/credit/sme/first-central", "/v1/credit/sme/premium") and country == "nigeria":
                    if "business_reg_no" not in params:
                        logger.error("Business registration number required for business credit insights")
                        return {"error": "Business registration number required for business credit insights"}
                    response = client.get(
                        f"{self.fastapi_url}{endpoint}",
                        params={"business_reg_no": params["business_reg_no"]},
                        headers=headers,
                        timeout=10.0
                    )
                elif endpoint in (
                    "/v1/credit/ke/individual",
                    "/v1/credit/ke/individual-premium",
                    "/v1/credit/ke/individual-mobileReport",
                    "/v1/credit/ke/individual-mobileReport-history",
                    "/v1/credit/ke/individual-variableReport",
                    "/v1/credit/ke/individual-variableScoreReport"
                ) and country == "kenya":
                    required_params = ["lastName", "firstName", "nationalID"]
                    if not all(param in params for param in required_params):
                        missing = [param for param in required_params if param not in params]
                        logger.error(f"Missing required parameters: {missing}")
                        return {"error": f"Missing required parameters: {missing}"}
                    response = client.get(
                        f"{self.fastapi_url}{endpoint}",
                        params={
                            "lastName": params["lastName"],
                            "firstName": params["firstName"],
                            "nationalID": params["nationalID"]
                        },
                        headers=headers,
                        timeout=10.0
                    )
                elif endpoint == "/v1/identity/ke/verifyIndividual" and country == "kenya":
                    required_params = ["id_type", "identifier", "first_name", "last_name"]
                    if not all(param in params for param in required_params):
                        missing = [param for param in required_params if param not in params]
                        logger.error(f"Missing required parameters: {missing}")
                        return {"error": f"Missing required parameters: {missing}"}
                    response = client.post(
                        f"{self.fastapi_url}{endpoint}",
                        json={
                            "id_type": params["id_type"],
                            "identifier": params["identifier"],
                            "first_name": params["first_name"],
                            "last_name": params["last_name"]
                        },
                        headers=headers,
                        timeout=10.0
                    )
                elif endpoint == "/v1/identity/ke/verifyBusiness" and country == "kenya":
                    if "registration_no" not in params:
                        logger.error("Registration number required for Kenya business verification")
                        return {"error": "Registration number required for Kenya business verification"}
                    response = client.post(
                        f"{self.fastapi_url}{endpoint}",
                        json={"registration_no": params["registration_no"]},
                        headers=headers,
                        timeout=10.0
                    )
                else:
                    logger.warning(f"Unsupported endpoint: {endpoint}")
                    return {"error": f"Endpoint {endpoint} not supported"}
                if response.headers.get("content-type") != "application/json":
                    logger.error(f"Unexpected response content type: {response.headers.get('content-type')}")
                    return {"error": f"Unexpected response content type: {response.headers.get('content-type')}"}
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed for {endpoint}: {e.response.status_code} - {e.response.text}")
            return {"error": f"API request failed: {e.response.status_code} - {e.response.text}"}
        except httpx.RequestError as e:
            logger.error(f"Network error during API call to {endpoint}: {e}")
            return {"error": f"Network error: {str(e)}"}
        except FileNotFoundError as e:
            logger.error(f"PDF statement file not found: {e}")
            return {"error": f"PDF statement file not found: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during API call to {endpoint}: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    def _code_generator_tool(self, language: str, task: str, framework: str = None) -> str:
        language = language.lower()
        framework = framework.lower() if framework else None
        
        # Map tasks to Kenya credit report endpoints
        task_to_endpoint = {
            "kenya_basic_report": "/v1/credit/ke/individual",
            "kenya_full_report": "/v1/credit/ke/individual-premium",
            "kenya_consumer_variable_report": "/v1/credit/ke/individual-variableReport",
            "kenya_consumer_variable_score_report": "/v1/credit/ke/individual-variableScoreReport",
            "kenya_mobile_loan_history_report": "/v1/credit/ke/individual-mobileReport-history",
            "kenya_mobile_loan_report": "/v1/credit/ke/individual-mobileReport"
        }
        
        if language == "nodejs" and task.lower() in task_to_endpoint:
            endpoint = task_to_endpoint[task.lower()]
            report_type = task.replace("kenya_", "").replace("_", " ").title()
            return f"""const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

app.get('/{task.lower().replace('_', '-')}', async (req, res) => {{
    const {{ firstName, lastName, nationalID }} = req.query;
    if (!firstName || !lastName || !nationalID) {{
        return res.status(400).json({{ error: 'firstName, lastName, and nationalID are required' }});
    }}

    try {{
        const response = await axios.get(
            'https://api.creditchek.africa{endpoint}',
            {{
                params: {{ firstName, lastName, nationalID }},
                headers: {{
                    Authorization: `Bearer ${{process.env.CREDITCHEK_API_KEY}}`,
                    'Content-Type': 'application/json'
                }}
            }}
        );
        res.json(response.data);
    }} catch (error) {{
        res.status(error.response?.status || 500).json({{ error: error.message }});
    }}
}});

app.listen(3000, () => console.log('{report_type} server running on port 3000'));"""
        
        if language == "python" and "webhook" in task.lower():
            if framework == "fastapi":
                return """import http.client.post
from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Any
import hmac
import hashlib

app = FastAPI()

def verify_webhook_signature(request_body: bytes, x_authenticity_signature: str, live_secret_key: str) -> bool:
    signature = hmac.new(live_secret_key.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, x_authenticity_signature)

@app.post("/webhook")
async def webhook_handler(request: Request):
    request_body = await request.body()
    x_authenticity_signature = request.headers.get("x-authenticity")
    live_secret_key = os.environ.get("CREDITCHEK_LIVE_SECRET_KEY")

    if not live_secret_key:
        raise HTTPException(status_code=500, detail="CREDITCHEK_LIVE_SECRET_KEY environment variable not set")

    if not verify_webhook_signature(request_body, x_authenticity_signature, live_secret_key):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = await request.json()
        print(f"Webhook received: {data}")
        return {"status": "OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""
           # elif platform == "django":
                return """import os
import json
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

def verify_webhook_signature(request_body: bytes, x_authenticity_signature: str, live_secret_key: str) -> bool:
    signature = hmac.new(live_secret_key.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, x_authenticity_signature)

@csrf_exempt
@require_POST
def webhook_handler(request):
    request_body = request.body
    x_authenticity_signature = request.META.get("HTTP_X_AUTH_SIGNATURE")
    live_secret_key = os.environ.get("CREDITCHEK_LIVE_SECRET_KEY")

    if not live_secret_key:
        return JsonResponse({"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}, status=500)

    if not verify_webhook_signature(request_body, x_authenticity_signature, live_secret_key):
        return JsonResponse({"error": "Invalid signature"}, status=401)

    try:
        data = json.loads(request_body)
        print(f"Webhook received: {data}")
        return HttpResponse("OK", status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)"""
            else:  # Default to Flask
                return """from flask import Flask, request, jsonify
import httpx
import os
import hmac
import hashlib

app = Flask(__name__)

def verify_webhook_signature(request_body, x_authenticity_signature, live_secret_key):
    signature = hmac.new(live_secret_key.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, x_authenticity_signature)

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    request_body = request.get_data()
    x_authenticity_signature = request.headers.get('x-authenticity')
    live_secret_key = os.environ.get('CREDITCHEK_LIVE_SECRET_KEY')

    if not live_secret_key:
        return jsonify({"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}), 500

    if not verify_webhook_signature(request_body, x_authenticity_signature, live_secret_key):
        return jsonify({"error": "Invalid signature"}), 401

    try:
        data = request.get_json()
        print(f"Webhook received: {data}")
        return "OK", 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500"""
                
        elif language == "python" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """import os
from creditchek import RecovaPRO

client = RecovaPRO(api_key=os.environ.get('CREDITCHEK_API_KEY'))

try:
    result = client.verify_identity(first_name='John', last_name='Doe', password='1980-01-01')
    print(f"Identification Verification Results: {result}")
except Exception as e:
    print(f"Error during identity verification: {e}")

try:
    schedule = client.get_repayment_schedule(customer_id='12345')
    print(f"Repayment Schedule: {schedule}")
except Exception as e:
    print(f"Error retrieving repayment schedule: {e}")"""
                
        elif language in ["javascript", "nodejs"] and "webhook" in task.lower():
            return """const express = require('express');
const crypto = require('crypto');
const app = express();

app.use(express.json());

function verifyWebhookSignature(requestBody, xAuthenticitySignature, liveSecretKey) {
    const signature = crypto.createHmac('sha256', liveSecretKey)
                           .update(JSON.stringify(requestBody))
                           .digest('hex');
    return signature === xAuthenticitySignature;
}

app.post('/webhook', (req, res) => {
    const xAuthenticitySignature = req.headers['x-authenticity'];
    const liveSecretKey = process.env.CREDITCHEK_LIVE_SECRET_KEY;

    if (!liveSecretKey) {
        return res.status(500).json({ error: 'CREDITCHEK_LIVE_SECRET_KEY environment variable not set' });
    }

    if (!verifyWebhookSignature(req.body, xAuthenticitySignature, liveSecretKey)) {
        return res.status(401).json({ error: 'Invalid signature' });
    }

    console.log('Webhook received:', req.body);
    res.status(200).send('OK');
});

app.listen(3000, () => console.log('Webhook server running on port 3000'));"""
                
        elif language in ["javascript", "nodejs"] and ("sdk" in task.lower() or "configure" in task.lower()):
            return """const RecovaPRO = require('recovapro');

const client = new RecovaPRO({ api_key: process.env.CREDITCHEK_API_KEY });

client.verifyIdentity({ firstName: 'John', lastName: 'Doe', dob: '1980-01-01' })
    .then(result => console.log(result))
    .catch(err => console.error(err));

client.getRepaymentSchedule({ customerId: '12345' })
    .then(schedule => console.log(schedule))
    .catch(err => console.error(err));"""
                
        elif language == "php" and "webhook" in task.lower():
            return """<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class WebhookController extends Controller
{
    public function handleWebhook(Request $request)
    {
        $requestBody = $request->getContent();
        $xAuthenticitySignature = $request->header('x-authenticity');
        $liveSecretKey = env('CREDITCHEK_LIVE_SECRET_KEY');

        if (!$liveSecretKey) {
            return response()->json(['error' => 'CREDITCHEK_LIVE_SECRET_KEY environment variable not set'], 500);
        }

        $signature = hash_hmac('sha256', $requestBody, $liveSecretKey);

        if (!hash_equals($signature, $xAuthenticitySignature)) {
            return response()->json(['error' => 'Invalid signature'], 401);
        }

        \Log::info('Webhook received: ' . json_encode($request->all()));
        return response('OK', 200);
    }
}"""
                
        elif language == "php" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """<?php

use CreditChek\RecovaPRO;

$client = new RecovaPRO(['api_key' => env('CREDITCHEK_API_KEY')]);

try {
    $result = $client->verifyIdentity([
        'first_name' => 'John',
        'last_name' => 'Doe',
        'dob' => '1980-01-01'
    ]);
    \Log::info('Verification result: ' . json_encode($result));
} catch (\Exception $e) {
    \Log::error('Error during identity verification: ' . $e->getMessage());
}

try {
    $schedule = $client->getRepaymentSchedule(['customer_id' => '12345']);
    \Log::info('Repayment schedule: ' . json_encode($schedule));
} catch (\Exception $e) {
    \Log::error('Error retrieving repayment schedule: ' . $e->getMessage());
}"""
                
        elif language == "go" and "webhook" in task.lower():
            return """package main

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "io/ioutil"
    "log"
    "net/http"
    "os"
)

func verifyWebhookSignature(requestBody []byte, xAuthenticitySignature, liveSecretKey string) bool {
    mac := hmac.NewItem(sha256.NewItem(), []byte(liveSecretKey))
    mac.Write(requestBody)
    expectedSignature := hex.EncodeToString(mac.Hash(nil))
    return hmac.compareDigest([]byte(expectedSignature), []byte(xAuthenticitySignature))
}

func webhookHandler(w http.ResponseWriter, r *http.Request) {
    requestBody, err := ioutil.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Error reading request body", http.StatusBadRequest)
        return
    }

    xAuthenticitySignature := r.Header.Get("x-authenticity")
    liveSecretKey := os.getenv("CREDITCHEK_LIVE_SECRET_KEY")

    if liveSecretKey == "" {
        http.Error(w, `{"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}`, http.StatusInternalServerError)
        return
    }

    if !verifyWebhookSignature(requestBody, xAuthenticitySignature, liveSecretKey) {
        http.Error(w, `{"error": "Invalid signature"}`, http.StatusUnauthorized)
        return
    }

    var payload map[string]interface{}
    json.Unmarshal(requestBody, &payload)
    log.Printf("Webhook received: %v", payload)
    w.Write([]byte("OK"))
}

func main() {
    http.HandleFunc("/webhook", webhookHandler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}"""
                
        elif language == "go" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """package main

import (
    "log"
    "os"
    "github.com/creditchek/recovapro"
)

func main() {
    client := recovapro.NewClient(os.getEnvironmentVariable("CREDITCHEK_API_KEY"))

    result, err := client.VerifyIdentity(recovapro.IdentityParams{
        FirstName: "John",
        LastName:  "Doe",
        DOB:       "1980-01-01",
    })
    if err != nil {
        log.Fatal("Verification failed: %v", err)
    }
    log.Printf("Verification result: %v", result)

    schedule, err := client.GetRepaymentSchedule("12345")
    if err != nil {
        log.Fatal("Repayment schedule failed: %v", err)
    }
    log.Printf("Repayment schedule: %v", schedule)
}"""
                
        return f"""// {language} implementation for {task}
// Framework: {framework or 'standard library'}

// This is a generic template. For {language}-specific implementation:
// 1. Consult the {language} documentation for HTTP server/client setup
// 2. Review CreditChek API documentation at https://docs.creditchek.africa
// 3. Adapt the patterns from our other language examples

// Key requirements:
// - Secure API key management
// - Proper error handling
// - HTTPS for all communications
// - Webhook signature verification (if applicable)

// For framework-specific guidance in {language}, consider:
// {', '.join(self.app_config["supported_frameworks"].get(language, ["Standard Library"]))}"""

    def _detect_language_and_framework(self, query: str) -> tuple[str, Optional[str]]:
        query_lower = query.lower()
        if self.forced_language:
            language = self.forced_language
            logger.debug(f"Using forced language: {language}")
        else:
            language = "NodeJS"
            for lang in self.app_config["supported_languages"]:
                if lang.lower() in query_lower or f"in {lang.lower()}" in query_lower:
                    language = lang
                    break

        framework = None
        supported_frameworks = self.app_config["supported_frameworks"].get(language, [])
        for fw in supported_frameworks:
            if fw.lower() in query_lower:
                framework = fw
                break
        if not framework and self.current_frameworks:
            framework = self.current_frameworks[0]

        logger.debug(f"Detected language: {language}, framework: {framework}")
        return language, framework
        
    def get_preconfigured_prompts(self) -> List[str]:
        try:
            prompts = self.app_config.get("preconfigured_prompts", [])
            logger.debug(f"Retrieved {len(prompts)} preconfigured prompts")
            return prompts
        except Exception as e:
            logger.error(f"Failed to retrieve preconfigured prompts: {e}")
            return []

    def select_tools(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower().strip()
        language, framework = self._detect_language_and_framework(query)
        country = "kenya" if "kenya" in query_lower else "nigeria"
        tools = []

        # --- Predefined prompt mappings ---
        prompt_mappings = {
            "how do i set up webhooks for creditchek transaction updates?": {
                "tool": "code_generator",
                "params": {"language": language, "framework": framework, "task": "webhook"},
                "description": f"Generate {language} webhook code using {framework or 'default framework'}"
            },
            "how do i get started with the creditchek apis?": {
                "tool": "document_retrieval",
                "params": {"query": "getting started with CreditChek APIs"},
                "description": "Retrieve getting started guide"
            },
            "what are the different service endpoints available in the creditchek apis?": {
                "tool": "document_retrieval",
                "params": {"query": f"all CreditChek API endpoints {country}"},
                "description": f"Retrieve list of available endpoints for {country}"
            },
            "what are the required headers and parameters for each endpoint?": {
                "tool": "document_retrieval",
                "params": {"query": f"CreditChek API headers parameters {country}"},
                "description": f"Retrieve headers and parameters for {country}"
            },
            "what are the available response formats for each endpoint?": {
                "tool": "document_retrieval",
                "params": {"query": f"CreditChek API response formats {country}"},
                "description": f"Retrieve response formats for {country}"
            },
            "how do i authenticate and generate an access token for the creditchek apis?": {
                "tool": "document_retrieval",
                "params": {"query": "CreditChek API authentication"},
                "description": "Retrieve authentication guide"
            },
            "are there any usage limits or rate limits for the creditchek apis?": {
                "tool": "document_retrieval",
                "params": {"query": "CreditChek API rate limits"},
                "description": "Retrieve rate limit information"
            },
            "how do i handle errors and interpret error codes in the creditchek apis?": {
                "tool": "document_retrieval",
                "params": {"query": "CreditChek API error handling"},
                "description": "Retrieve error handling guide"
            },
            "how can i test the creditchek apis in a sandbox environment before going live?": {
                "tool": "document_retrieval",
                "params": {"query": "CreditChek API sandbox testing"},
                "description": "Retrieve sandbox testing guide"
            }
        }

        # --- Check preconfigured prompt match ---
        if query_lower in [p.lower() for p in self.app_config.get("preconfigured_prompts", [])]:
            for prompt, config in prompt_mappings.items():
                if query_lower == prompt.lower():
                    tools.extend(config if isinstance(config, list) else [config])
                    logger.debug(f"Matched preconfigured prompt: {prompt}")
                    return tools
        if "income insight" in query_lower or "process income" in query_lower:
            tools.append({
                "tool": "_creditchek_api",
                "params": {
                    "endpoint": "/v1/income/api/process-income",
                    "params": {
                        "bvn": "12345678901",
                        "accountName": "John Doe",
                        "accountNumber": "1234567890",
                        "accountType": "savings",
                        "bankName": "Access Bank",
                        "bankCode": "044",
                        "statement": "test_data/sample_statement.pdf"  # leave this as is or skip if simulating
                    },
                    "country": "nigeria"
                },
                "description": "Call /v1/income/api/process-income to process PDF bank statement for income insight"
            })
            return tools


        # --- Dynamic report matching for Kenya ---
        if country == "kenya" and any(
            kw in query_lower for kw in [
                "credit insight", "credit history", "mobile loan report",
                "consumer variable report", "basic report", "full report",
                "consumer variable report with score", "mobile loan history report"
            ]
        ):
            endpoint_map = {
                "basic report": "/v1/credit/ke/individual",
                "full report": "/v1/credit/ke/individual-premium",
                "mobile loan report": "/v1/credit/ke/individual-mobileReport",
                "mobile loan history report": "/v1/credit/ke/individual-mobileReport-history",
                "consumer variable report": "/v1/credit/ke/individual-variableReport",
                "consumer variable report with score": "/v1/credit/ke/individual-variableScoreReport"
            }

            matched_key, endpoint = next(
                ((k, ep) for k, ep in endpoint_map.items() if k in query_lower),
                ("consumer variable report", "/v1/credit/ke/individual-variableReport")
            )

            params = {
                "firstName": "Jane",
                "lastName": "Doe",
                "nationalID": "550000055"
            }
            tools.extend([
                {
                    "tool": "_creditchek_api",
                    "params": {"endpoint": endpoint, "params": params, "country": country},
                    "description": f"Call {endpoint} API for Kenya {matched_key}"
                },
                {
                    "tool": "code_generator",
                    "params": {"language": language, "framework": framework, "task": f"kenya_{matched_key.replace(' ', '_')}"},
                    "description": f"Generate {language} code for Kenya {matched_key}"
                },
                {
                    "tool": "document_retrieval",
                    "params": {"query": f"{matched_key} Kenya"},
                    "description": f"Retrieve {matched_key} documentation for Kenya"
                }
            ])
            return tools
        if any(kw in query_lower for kw in ["verify a business", "verify business", "business verification", "verify company"]):
            if country == "kenya":
                tools.append({
                    "tool": "_creditchek_api",
                    "params": {
                        "endpoint": "/v1/identity/ke/verifyBusiness",
                        "params": {"registration_no": "CPR/2015/123456"},
                        "country": country
                    },
                    "description": "Call /v1/identity/ke/verifyBusiness API for Kenya business verification"
                })
                return tools

        # --- Business credit insights (Nigeria only) ---
        if any(k in query_lower for k in ["business", "sme"]):
            if country == "nigeria":
                endpoint = (
                    "/v1/credit/sme/crc" if "crc" in query_lower else
                    "/v1/credit/sme/first-central" if "first-central" in query_lower else
                    "/v1/credit/sme/premium"
                )
                tools.append({
                    "tool": "_creditchek_api",
                    "params": {"endpoint": endpoint, "params": {"business_reg_no": "RC123456"}, "country": country},
                    "description": f"Call {endpoint} API for Nigeria business credit insights"
                })
                return tools

        # --- Individual credit insight APIs (Nigeria only) ---
        individual_endpoint_map = {
            "crc": "/v1/credit/crc",
            "credit-registry": "/v1/credit/credit-registry",
            "first-central": "/v1/credit/first-central",
            "advanced": "/v1/credit/advanced",
            "premium": "/v1/credit/premium"
        }

        for keyword, endpoint in individual_endpoint_map.items():
            if keyword in query_lower:
                tools.extend([
                    {
                        "tool": "_creditchek_api",
                        "params": {"endpoint": endpoint, "params": {"bvn": "12345678901"}, "country": country},
                        "description": f"Call {endpoint} API for Nigeria individual credit insights"
                    },
                    {
                        "tool": "document_retrieval",
                        "params": {"query": f"credit insights {country}"},
                        "description": f"Retrieve credit insights documentation for {country}"
                    }
                ])
                return tools

        # --- Identity verification ---
        if "verify individual" in query_lower or "individual verification" in query_lower:
            if country == "kenya":
                tools.append({
                    "tool": "_creditchek_api",
                    "params": {
                        "endpoint": "/v1/identity/ke/verifyIndividual",
                        "params": {
                            "id_type": "national_id",
                            "identifier": "550000055",
                            "first_name": "Jane",
                            "last_name": "Doe"
                        },
                        "country": country
                    },
                    "description": "Call /v1/identity/ke/verifyIndividual API for Kenya individual verification"
                })
        else:
                # Nigeria: support multiple ID types
            id_type = "nin"
            id_param_name = "nin"
            id_value = "12345678901"  # default fallback

            if "bvn" in query_lower:
                id_type = "bvn"
                id_param_name = "bvn"
                id_value = "12345678901"
            elif "cac" in query_lower:
                id_type = "cac"
                id_param_name = "cac_number"
                id_value = "RC123456"
            elif "bank account" in query_lower or "bankaccount" in query_lower:
                id_type = "bank_account"
                id_param_name = None
                id_value = None
                tool_params = {
                    "type": "bank_account",
                    "account_number": "1234567890",
                    "bank_code": "044"
                }
            elif "international passport" in query_lower or "passport" in query_lower:
                id_type = "international_passport"
                id_param_name = "passport"
                id_value = "A1234567"
            elif "driver's license" in query_lower or "drivers license" in query_lower:
                id_type = "driver_license"
                id_param_name = "driver_license"
                id_value = "DL1234567"

            params = {"type": id_type}
            if id_param_name:
                params[id_param_name] = id_value
            elif id_type == "bank_account":
                params.update(tool_params)

            tools.append({
                "tool": "_creditchek_api",
                "params": {
                    "endpoint": "/api/v1/identity/verifyData",
                    "params": params,
                    "country": "nigeria"
                },
                "description": f"Call /api/v1/identity/verifyData API for Nigeria individual verification using {id_type.upper()}"
            })

            return tools if tools else []

    def generate_response(self, query: str) -> Dict[str, Any]:
        logger.info(f"Processing query: {query}")
        query_lower = query.lower()
        language, framework = self._detect_language_and_framework(query)
        logger.debug(f"Detected language: {language}, framework: {framework}")

        history = self.memory.load_memory_variables({}).get("chat_history", [])
        if history and not any("webhook" in str(msg).lower() or "sdk" in str(msg).lower() for msg in history):
            logger.debug("Clearing irrelevant chat history")
            self.memory.clear()

        if any(f"show {lang.lower()} example" in query_lower for lang in self.app_config["supported_languages"]):
            task = self._determine_task(query_lower)
            try:
                code = self.code_generator_tool(language, task, framework)
                response = self._build_code_response(query, language, framework, task, code)
                logger.info("Code generation response generated")
                return response
            except Exception as e:
                logger.error(f"Code generation failed for {language}/{framework}: {e}")
                return self.fallback_response(query, language, framework)
        try:
            history = self.memory.load_memory_variables({}).get("chat_history", [])
            if not isinstance(history, list):
                history = []
        except Exception as e:
            logger.warning(f"Chat history retrieval failed: {e}")
            history = []

        try:
            plan, results = self._execute_tools(query, language, framework)
            
            if not plan or "steps" not in plan or not isinstance(plan["steps"], list):
             raise ValueError("Plan is invalid or empty")
            # Extract hostname from the configured FastAPI base URL
            parsed = urlparse(self.fastapi_url)
            hostname = "api.creditchek.africa"

            tool_context_lines = []
            for step in plan["steps"]:
                tool = step.get("tool")
                ep = step["params"].get("endpoint")
                body = step["params"].get("params", {})
                tool_context_lines.append(
                    f"Tool: {tool}\n"
                    f"Host: {hostname}\n"
                    f"Endpoint: {ep}\n"
                    f"Method: POST\n"
                    f"Body:\n{json.dumps(body, indent=2)}"
                )


            context = "\n\n".join(tool_context_lines)

            final_prompt = PromptTemplate.from_template(
                """You are Kasi, an AI assistant for CreditChek API integration.
        Use the tool results to answer the query in {language} using {framework}.
        Provide code examples and suggest next steps.
        Tool Results: {context}
        Query: {query}
        Chat History: {chat_history}
        Answer in markdown format with sections:

        Framework Choice: Explain why {framework} was chosen.
        Steps: Step-by-step guide to address the query using {framework}.
        Example: Code example in {language} using {framework} (use {language}\n...\n).
        Additional Notes: Clarifications, including alternative frameworks.
        Next Steps: Proactive suggestions for related tasks. Ensure Next Steps are specific and actionable.""").partial(
                language=language.lower(),
                framework=framework or "default framework"
            )

            logger.info("Invoking LLM for final response")
            logger.debug("Final Prompt:\n" + final_prompt.format(
                context=context,
                query=query,
                chat_history=history
            ))
        
            final_response = self.llm.invoke(final_prompt.format(
                context=context,
                query=query,
                chat_history=history
            ))

            # Append tool info (e.g. if statement file was missing)
            tool_info_messages = [
                result["output"]["info"]
                for result in results
                if isinstance(result["output"], dict) and "info" in result["output"]
            ]

            if tool_info_messages:
                info_section = "\n\n📘 **CreditChek API Info:**\n" + "\n".join(tool_info_messages)
                final_response.content += info_section

            response = {
                "plan": plan,
                "response": {
                    "text": final_response.content,
                    "code_snippets": {
                        result["tool"]: result["output"]
                        for result in results if result["tool"] == "code_generator"
                    }
                },
                "tool_results": results
            }

            self.memory.save_context({"question": query}, {"answer": response["response"]["text"]})
            logger.info("Agentic response generated successfully")
            return response

        except Exception as e:
            logger.error(f"Agentic loop failed: {e}")
            return self.fallback_response(query, language, framework)

    def _determine_task(self, query_lower: str) -> str:
        if "webhook" in query_lower:
            return "webhook"
        elif "sdk" in query_lower:
            return "sdk"
        elif any(k in query_lower for k in ["basic", "full", "consumer variable", "mobile"]):
            return f"kenya_{query_lower.split()[1].replace('-', '_')}report"
        else:
            return "sdk"

    def _build_code_response(self, query: str, language: str, framework: str, task: str, code: str) -> Dict[str, Any]:
        response = {
            "plan": {
                "steps": [
                    {
                        "tool": "code_generator",
                        "params": {"language": language, "framework": framework, "task": task},
                        "description": f"Generate {language} code for {task} using {framework or 'default framework'}"
                    }
                ]
            },
            "response": {
                "text": f"Here's a {language} example for CreditChek API {task.replace('kenya', '').replace('', ' ').title()} using {framework or 'default framework'}:",
                "code_snippets": {language: code}
            },
            "tool_results": [{"tool": "code_generator", "output": code}]
        }
        self.memory.save_context({"question": query}, {"answer": response["response"]["text"]})
        return response

    def _execute_tools(self, query: str, language: str, framework: str):
        suggested_tools = self.select_tools(query)
        plan = {"steps": suggested_tools}
        results = []
        for step in suggested_tools:
            tool = step.get("tool")
            params = step.get("params", {})
            try:
                if tool == "_creditchek_api":
                    output = self._creditchek_api_tool(
                        params.get("endpoint"),
                        params.get("params"),
                        params.get("country", "nigeria")
                    )
                elif tool == "document_retrieval":
                    output = [doc.page_content for doc in self._document_retrieval_tool(params.get("query", query))]
                elif tool == "code_generator":
                    output = self._code_generator_tool(
                        params.get("language", language),
                        params.get("task", query),
                        params.get("framework", framework)
                    )
                else:
                    output = f"Error: Unknown tool {tool}"
                    logger.warning(output)
                results.append({"tool": tool, "output": output})
            except Exception as e:
                logger.error(f"Tool {tool} failed: {e}")
                results.append({"tool": tool, "output": f"Error: {str(e)}"})
        return plan, results

    def fallback_response(self, query: str, language: str = "NodeJS", framework: str = None) -> Dict[str, Any]:
        logger.info(f"Executing fallback response for query: {query} in {language}/{framework or 'default framework'}")
        try:
            docs = self._document_retrieval_tool(query)
            context = "\n".join(doc.page_content for doc in docs)

            code_snippet = {}
            code_text = f"No {language} code example available"
            task = None

            if any(keyword in query.lower() for keyword in [
                "basic report", "full report", "consumer variable report",
                "consumer variable score report", "mobile loan history report", "mobile loan report"]):

                task = f"kenya{''.join(query.lower().split()[1:3]).replace('with_score', 'score')}_report"
                code = self._code_generator_tool(language, task, framework)
                code_snippet = {language: code}
                code_text = code
            else:
                task = "sdk"
            try:
                history = self.memory.load_memory_variables({}).get("chat_history", [])
                if not isinstance(history, list):
                    history = []
            except Exception as e:
                logger.warning(f"Chat history retrieval failed: {e}")
                history = []

            qa_prompt = PromptTemplate.from_template(
                """You are Kasi, an AI assistant for CreditChek API integration.
Context: {context}
Question: {query}
Provide a detailed answer with steps, a {language} code example using {framework}, and next steps.

DO NOT say “replace with actual endpoint” if the endpoint and host are already specified. 
Assume the tool result gives the correct endpoint and host.

Tool Results: {context}
Query: {query}
Chat History: {chat_history}

Answer in markdown format with sections:

Framework Choice: Explain why {framework} was chosen.
Steps: Step-by-step guide to address the query using {framework}.
Example: Code example in {language} using {framework} (use {language}\n...\n).
Additional Notes: Clarifications, including alternative frameworks.
Next Steps: Proactive suggestions for related tasks. Include the following code example in the Example section: {code_text} Answer:"""
            ).partial(language=language.lower(), framework=framework or "default framework")

            response = self.llm.invoke(qa_prompt.format(
                context=context, query=query, code_text=code_text
            ))
            logger.debug(f"Fallback LLM response: {response.content}")

            result = {
                "plan": {
                    "steps": [
                        {"tool": "document_retrieval", "params": {"query": query}, "description": "Retrieve documents"},
                        {"tool": "code_generator", "params": {"language": language, "framework": framework, "task": task}, "description": f"Generate {language} {task} example"} if code_snippet else {}
                    ]
                },
                "response": {
                    "text": response.content,
                    "code_snippets": code_snippet
                },
                "tool_results": [
                    {"tool": "document_retrieval", "output": [doc.page_content for doc in docs]},
                    {"tool": "code_generator", "output": code_snippet.get(language, "")} if code_snippet else {}
                ]
            }
            self.memory.save_context({"question": query}, {"answer": result["response"]["text"]})
            logger.info("Fallback response generated")
            return result

        except Exception as e:
            logger.error(f"Fallback response failed: {e}")
            return {
                "plan": {"steps": []},
                "response": {
                    "text": f"Error: Unable to generate response for '{query}'. Please check logs and ensure GOOGLE_API_KEY is valid.",
                    "code_snippets": {}
                },
                "tool_results": []
            }

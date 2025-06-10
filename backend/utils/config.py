import os
from dotenv import load_dotenv
import logging
import streamlit as st

logger = logging.getLogger(__name__)
load_dotenv()

def get_pinecone_config():
    config = {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", "us-east1-aws"),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
        "namespace": os.getenv("PINECONE_NAMESPACE", None)
    }
    logger.info(f"get_pinecone_config output: {config | {'api_key': '***'}}")
    return config

def get_openai_config():
    return {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "model": "gemini-2.0-flash",
        "temperature": 0.7
    }

def get_app_config():
    return {
        "doc_url": "https://docs.creditchek.africa",
        "supported_frameworks": {
            "Python": ["Flask", "FastAPI", "Django"],
            "JavaScript": ["Express", "Koa", "NestJS"],
            "NodeJS": ["Express"],
            "PHP": ["Laravel", "Slim"],
            "Go": ["net/http", "Gin", "Echo"],
            "Java": ["Spring", "Jakarta EE"],
            "C#": [".NET", "ASP.NET Core"],
            "Ruby": ["Rails", "Sinatra"],
            "Rust": ["Actix", "Rocket"],
            "Swift": ["Vapor", "Perfect"],
            "Kotlin": ["Ktor", "Spring Boot"],
            "Other": ["Standard Library"]
        },
        "supported_languages": ["Python", "JavaScript", "NodeJS", "PHP", "Go", "Java", "C#", "Ruby", "Rust", "Swift", "Kotlin", "Other"],
        "default_language": os.getenv("DEFAULT_LANGUAGE", "NodeJS"),
        "preconfigured_prompts": [
            "How do I set up webhooks for CreditChek transaction updates?",
            "How do I get started with the CreditChek APIs?",
            "What are the different service endpoints available in the CreditChek APIs?",
            "What are the required headers and parameters for each endpoint?",
            "What are the available response formats for each endpoint?",
            "How do I authenticate and generate an access token for the CreditChek APIs?",
            "Are there any usage limits or rate limits for the CreditChek APIs?",
            "How do I handle errors and interpret error codes in the CreditChek APIs?",
            "How can I test the CreditChek APIs in a sandbox environment before going live?"
        ],
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

def setup_page_config():
    st.set_page_config(
        page_title="Kasi - CreditChek API Assistant",
        page_icon=":robot:",
        layout="wide"
    )
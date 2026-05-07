import os
import logging

logger = logging.getLogger(__name__)


PLACEHOLDER_KEY = "PLACEHOLDER"

'''
OpenAI
'''
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', PLACEHOLDER_KEY)
if OPENAI_API_KEY == PLACEHOLDER_KEY:
    print("WARNING: OPENAI_API_KEY not set, using placeholder key")

def enable_openai(api_key: str):
    global OPENAI_API_KEY
    OPENAI_API_KEY = api_key


'''
Google
'''
GOOGLE_GENAI_API_KEY = os.getenv('GOOGLE_GENAI_API_KEY', PLACEHOLDER_KEY)
if GOOGLE_GENAI_API_KEY == PLACEHOLDER_KEY:
    print("WARNING: GOOGLE_GENAI_API_KEY not set, using placeholder key")

def enable_google_genai(api_key: str):
    global GOOGLE_GENAI_API_KEY
    GOOGLE_GENAI_API_KEY = api_key


'''
Together
'''
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY', PLACEHOLDER_KEY)
if TOGETHER_API_KEY == PLACEHOLDER_KEY:
    print("WARNING: TOGETHER_API_KEY not set, using placeholder key")

def enable_together(api_key: str):
    global TOGETHER_API_KEY
    TOGETHER_API_KEY = api_key

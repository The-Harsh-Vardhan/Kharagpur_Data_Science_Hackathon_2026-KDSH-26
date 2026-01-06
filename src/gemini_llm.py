import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
# API key is loaded from .env file
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please set it in your .env file.\n"
        "Copy .env.example to .env and add your API key."
    )

genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-pro')


def gemini_call(prompt):
    """
    Calls Google Gemini API with the provided prompt.
    Returns the response text.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Fallback to NEUTRAL if API call fails
        return "NEUTRAL"

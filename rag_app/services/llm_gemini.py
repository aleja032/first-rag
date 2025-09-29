import settings
from langchain.chat_models import init_chat_model

    
llm = init_chat_model("google_genai:gemini-2.0-flash")
import os
import getpass
from langchain_community.llms import Ollama
from openai import OpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

def llama(prompt, model):
    """
    llama(prompt, "llama3")
    """
    llm = Ollama(model=model)
    res = llm.invoke(prompt)
    return res


def gpt(prompt, model):
    """
    gpt(prompt, "gpt-3.5-turbo")
    """
    client = OpenAI()
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content
    
def gemini(prompt, model):
    """
    gemini(prompt, "gemini-pro")
    """
    llm = ChatGoogleGenerativeAI(model=model)
    result = llm.invoke(prompt)
    return result.content


def claude(prompt, model):
    """
    claude(prompt, "claude-3-opus-20240229")
    """
    llm = ChatAnthropic(model_name=model)
    response = llm.invoke(prompt)
    return response.content
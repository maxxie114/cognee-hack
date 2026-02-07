# Ensure your dependencies are installed with:
# pip install openai weave
# (or: uv add openai weave)

# Find your OpenAI API key at: https://platform.openai.com/api-keys
# Ensure that your OpenAI API key is available in .env or:
# os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"

import os
import weave
from openai import OpenAI

# Load .env so OPENAI_API_KEY is available
from dotenv import load_dotenv
load_dotenv()

weave.init("your-username/weave-demo")  # Set your Weave project (entity/project)


@weave.op  # 🐝 Decorator to track requests
def create_completion(message: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # gpt-5 not available; using gpt-4o-mini
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    message = "Tell me a joke."
    result = create_completion(message)
    print(result)

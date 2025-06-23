import os
import enum
from openai import OpenAI

class Models(enum.Enum):
    GEMINI_PRO_MODEL = "models/gemini-2.5-pro-preview-05-06"
    GEMINI_FLASH_MODEL = "models/gemini-2.5-flash-preview-04-17-thinking"

os.environ["OPENAI_API_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
def get_completion(messages, model="gemini-2.0-flash", temperature=0, is_chat=True):
    try:
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_URL"],
        )

        if is_chat:
            # for chat models, we need to set the role to "user"
            response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
            )
        else:
            response = client.completions.create(
                model=model,
                prompt=messages
            )    

        print(f"Chat response: {response}")
        return response.choices[0].message.content or "No"
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Error: {e}"
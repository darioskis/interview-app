# interview-app

AI assistant to prepare you to crush your next interview.

## Setup

1. Create a Python virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-your-key
```

3. Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

## Features

- Collects job description and CV/resume snippets to extract key requirements, strengths, and weaknesses.
- Displays insights, match score, and likelihood of success on the left panel.
- Provides a chat assistant on the right panel to practice concise interview answers using a low-temperature LLM.

## Customizing the chat system message

The chat assistant always receives a system instruction so responses stay concise and grounded
in the candidate profile. You can edit the wording of this instruction (or replace it with your
own) in `backend/llm_service.py` inside `LLMService.chat_response`, where the `system_prompt`
variable is defined and passed to the OpenAI Responses API.

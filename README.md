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

## Customizing prompts

All LLM prompts live in `backend/prompts/` so you can edit behavior without touching Python code. Each file uses `{placeholder}` tokens that are filled at runtime:

- `extract_requirements.txt` – job description analysis
- `strengths_weaknesses.txt` – CV alignment highlights
- `match_report.txt` – match score JSON request
- `chat_system.txt` – system instruction applied to every chat reply
- `chat_prompt.txt` – wraps the conversation and profile context

If a prompt file is missing, the app falls back to a built-in default and logs a warning.

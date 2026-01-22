import subprocess

def generate_answer(prompt: str) -> str:
    """
    Generates answer using a local LLM via Ollama.
    """
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

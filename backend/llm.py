import subprocess

# Small model for low-RAM systems
MODEL_NAME = "tinyllama:latest"


def generate_answer(prompt: str) -> str:
    """
    Generate answer using local Ollama LLM.
    Handles Windows stdout/stderr issues safely.
    """
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True
    )

    output = result.stdout.strip()
    if not output:
        output = result.stderr.strip()

    return output

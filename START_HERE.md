# Your RAG project is ready

Start with [README.md](README.md). It contains setup instructions for Windows, macOS/Linux, Ollama, OpenAI, and Docker.

The fastest preview, after installing Python 3.11–3.13:

```bash
python -m pip install -r requirements.txt
python -m grounded ingest samples
python -m grounded serve
```

Open http://127.0.0.1:8000.

The default preview is extractive, not an LLM. Follow the README's **Run with a real local LLM** section to enable Ollama, or use the OpenAI setup with your API key.

Before putting it on your resume:

1. Run it with a real LLM and your own documents.
2. Read [the architecture notes](docs/ARCHITECTURE.md).
3. Use [the resume and interview guide](docs/RESUME_AND_INTERVIEW.md).
4. Review [what was actually validated](docs/VALIDATION.md).

The project includes 42 passing core/API tests and a measured demo benchmark. Live LLM inference, Docker, and browser automation require verification on your machine.

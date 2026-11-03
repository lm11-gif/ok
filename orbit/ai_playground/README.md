# Orbit — AI Playground

Orbit is a local-first browser workspace for experimenting with AI models and reusable agents.

## Run it

1. Open `index.html` in a modern desktop browser.
2. Start in Demo mode, or open **Settings** to connect a model.

No build tools, Node.js, or installation are required for the browser app itself.

## Connect a model

### Ollama
- Start Ollama on your computer and pull a model.
- In Orbit → Settings, choose **Ollama — local model**.
- Use `http://localhost:11434` as the base URL and enter the exact model tag.
- Use **Find models** or **Test connection**.

### OpenAI-compatible endpoint
- In Settings, choose **OpenAI-compatible API**.
- Enter the provider's API base URL (usually ending in `/v1`), model name, and API key if required.
- The endpoint must support `GET /models` and `POST /chat/completions`.

## Features

- Chat lab with reusable agent instructions and conversation history.
- Agent forge to create, edit, and reuse distinct AI personas.
- Model arena to send one prompt to two model IDs on the same endpoint.
- Prompt studio with saved system prompts and a test runner.
- Experiments log with export.
- Memory vault that attaches saved notes to future chat requests.
- Tool bench: JSON formatter, text/token estimator, simple text diff, and arithmetic calculator.
- Workspace backup as JSON.

## Notes and limitations

- Demo mode is simulated and does not call a real model.
- Orbit is a browser front end, not an AI model or agent runtime. The connected model must be running and accessible.
- Browser CORS rules can block requests to local or hosted endpoints.
- API keys and workspace information are stored in this browser's local storage. Do not use this on a shared computer, and do not put secrets in the Memory Vault.
- The token estimator is approximate.
- Chat responses use non-streaming requests in this first release.

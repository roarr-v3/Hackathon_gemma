# Gemma Personal Compute Mesh

## One-line pitch

A private voice assistant whose iPhone captures and speaks locally, while a trusted
Mac or home compute node runs heavier Gemma inference, private document retrieval,
and personal LoRA adapters—without sending personal data to a cloud AI provider.

## Platform target

The hackathon implementation targets iPhone on iOS 27 and a macOS 27/Xcode 27
development and compute environment. It is intentionally optimized for recent
Pro-class iPhones, including iPhone 16 Pro and iPhone 17 Pro. Supporting older iOS
versions, iPad, or memory-constrained legacy devices is out of scope.

## Official hackathon track

**Primary submission: Track 1 — Edge / On-Device**

The core experience begins and ends on the iPhone:

- speech is transcribed locally;
- recordings, transcripts, documents, and conversations belong to the phone;
- Gemma 4 E2B can already run fully on the phone through Cactus;
- the app continues to work when the Mac is unavailable;
- the phone can privately hand off expensive work to a user-owned compute node;
- answers are displayed and spoken locally.

Context engineering and adapter training strengthen the project, but the submission
should tell one simple Track 1 story: **local-first AI that expands to nearby,
user-owned compute without becoming cloud AI.**

## Problem

Voice assistants are convenient, but personal speech, documents, and questions are
often processed by third-party cloud systems. Fully on-device models improve privacy
and offline access, but a phone has limited memory, battery, and training capacity.

Users should not have to choose between privacy and capability.

## Solution

The system forms a small personal AI ecosystem from two trusted devices:

1. The iPhone is the private interface and source of truth.
2. A local Gemma model on the iPhone provides an offline fallback.
3. A Mac or local compute node runs Gemma 4 E4B for higher-capability inference.
4. Documents selected on the phone are transferred only to that paired node.
5. The node builds a private retrieval index from those documents.
6. The node can train a lightweight LoRA adapter for personal terminology, style,
   and task behavior.
7. Each request uses the best available execution path.
8. The response returns to the phone, appears in the conversation, and is read aloud
   using local text-to-speech.

## Existing foundation

The project in `/Users/sinan/Documents/cactus` already provides:

- a SwiftUI iPhone chat interface;
- Gemma 4 E2B local inference through `cactus-ios.xcframework`;
- private inference with cloud handoff disabled;
- conversation state and model reset/unload support;
- model download, checksum verification, extraction, storage, and deletion;
- support for a roughly 4 GB quantized model on a physical iPhone.

The hackathon implementation should extend this working app rather than replace it.
The intended capability ladder is Gemma 4 E2B through Cactus on the iPhone and a
4-bit MLX conversion of Gemma 4 E4B through vLLM-Metal on the Mac. A locally
installed 12B Q4 GGUF model may be served by LM Studio as an optional quality mode,
but is not the memory-safe demo default.

## End-to-end user experience

### First-time pairing

1. The user starts the companion server on the Mac.
2. The iPhone discovers it on the local network using Bonjour.
3. Both screens show a short pairing code.
4. After confirmation, the phone stores the node identity and authentication token
   in the Keychain.
5. The app displays `Phone`, `Mac`, or `Offline` as the active compute route.

### Voice question

1. The user presses and holds the microphone button.
2. The iPhone captures audio and performs local speech-to-text.
3. The transcript is shown and saved on the phone.
4. A router chooses:
   - on-device Cactus inference when no document is attached;
   - Mac inference whenever one or more documents are attached.
   - If a document is attached and the Mac is unavailable, the request stops with a
     visible error instead of silently answering without the document.
5. For Mac inference, the phone sends the transcript, selected workspace, and
   conversation identifier to the companion API.
6. The server retrieves relevant document chunks and sends a structured prompt to
   Gemma through the selected LoRA adapter.
7. The answer streams back to the phone.
8. The app displays the answer and reads it aloud locally.

### Personalizing with documents

1. The user chooses one or more files in the iPhone document picker.
2. The app shows exactly which files will leave the phone.
3. The phone uploads the files to the paired Mac over the local network.
4. The server extracts text, chunks it, and builds a private retrieval index.
5. The workspace becomes immediately usable through retrieval.
6. Optionally, the user taps `Train Personal Adapter`.
7. The server creates training examples for terminology, style, and desired tasks,
   trains a LoRA adapter, evaluates it, and registers it with the inference backend.
8. Training progress is sent to the phone.
9. When ready, the adapter appears in the workspace selector.

## Why retrieval and LoRA are both needed

LoRA should not be presented as a database for arbitrary document facts.

- **Retrieval** supplies exact, current facts and evidence from the uploaded files.
- **LoRA** teaches stable preferences: vocabulary, formatting, tone, domain task
  patterns, and how to use the retrieved context.
- **Gemma** reasons over the question, conversation, retrieved evidence, and learned
  adapter behavior.

For the hackathon demo, document retrieval is the reliable path. LoRA training is the
visible personalization path.

## Architecture

```text
┌──────────────────────────── iPhone ─────────────────────────────┐
│                                                                 │
│  Microphone → local STT → transcript store → request router     │
│                                      │                          │
│                   ┌──────────────────┴──────────────────┐       │
│                   │                                     │       │
│          Cactus + Gemma 4 E2B                    Paired-node API │
│          offline/local fallback                  client          │
│                   │                                     │       │
│                   └──────────────────┬──────────────────┘       │
│                                      ↓                          │
│                          answer store → local TTS                │
│                                                                 │
└──────────────────────────────────────┬──────────────────────────┘
                                       │ trusted LAN
                                       │ authenticated requests
┌──────────────────────── Mac / compute node ─────────────────────┐
│                                                                 │
│  FastAPI companion service                                      │
│       ├── pairing and health                                    │
│       ├── document ingestion → chunks → private retrieval index │
│       ├── training jobs → PEFT/LoRA → adapter registry          │
│       └── chat orchestration                                    │
│                  ├── retrieve evidence                          │
│                  ├── choose base model or adapter               │
│                  └── vLLM-compatible inference backend          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### iPhone application

#### `SpeechInputService`

- Requests microphone and speech-recognition permission.
- Captures live audio.
- Produces partial and final transcripts.
- Uses Apple's on-device Speech APIs where available.
- Persists the final transcript before sending a network request.

#### `SpeechOutputService`

- Owns a retained `AVSpeechSynthesizer`.
- Speaks complete responses or sentence-sized streamed chunks.
- Supports pause, resume, stop, voice selection, and speaking rate.

#### `ConversationStore`

- Saves conversations and messages locally.
- Records which route produced each answer: `phone` or `node`.
- Stores document/workspace references, not silent background uploads.
- MVP storage can be a JSON file; SwiftData is the better follow-up.

#### `ComputeNodeClient`

- Discovers `_gemma-mesh._tcp` services with `NWBrowser`.
- Performs pairing and stores credentials in Keychain.
- Calls the REST endpoints.
- Consumes server-sent events for streamed answers and job progress.

#### `InferenceRouter`

Input:

- network availability;
- paired-node health;
- user privacy mode;
- request type;
- selected workspace/adapter.

Output:

- `onDevice`, `pairedNode`, or a visible failure state.

It must never silently send data off-device. The UI always shows the selected route.

#### `DocumentWorkspaceView`

- Imports PDF, text, Markdown, and document files.
- Shows upload consent, progress, processing status, and errors.
- Lists indexed documents.
- Starts an optional adapter-training job.
- Shows `Queued → Preparing → Training → Evaluating → Ready`.

### Mac companion service

#### API gateway

Use FastAPI for the hackathon:

- small implementation surface;
- automatic API schema;
- simple file uploads;
- straightforward streaming and job-status endpoints.

#### Document pipeline

1. Validate file type and size.
2. Save under a generated workspace identifier.
3. Extract text.
4. Normalize and split into overlapping chunks.
5. Create embeddings locally.
6. Store chunks and metadata in a local vector index.
7. Return ingestion status to the phone.

For a fast MVP, use SQLite plus a lightweight local vector index. Every retrieved
chunk retains `document_name`, `page`, and `chunk_id` for citations.

#### LoRA training pipeline

1. Build instruction examples from the uploaded material.
2. Keep a deterministic train/evaluation split.
3. Train a PEFT LoRA adapter against the exact same base-model revision used for
   inference.
4. Save `adapter_config.json`, adapter weights, training metadata, and evaluation
   results.
5. Register the adapter only if it passes basic validation.
6. Expose the adapter as a selectable model to the chat orchestrator.

Suggested initial configuration:

- rank: 8 or 16;
- alpha: 16 or 32;
- dropout: 0.05;
- 1–3 epochs;
- conservative learning rate;
- small, visible training dataset;
- fixed seed for a repeatable demo.

The exact target modules must be validated against the Gemma 4 implementation used
by the training stack.

#### Inference backend

Define a small backend interface:

```python
class InferenceBackend(Protocol):
    async def health(self) -> bool: ...
    async def models(self) -> list[str]: ...
    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
    ) -> AsyncIterator[str]: ...
```

Implement:

- `VLLMBackend` for an NVIDIA/Linux or otherwise supported vLLM node;
- optionally `MLXBackend` for direct Apple Silicon execution;
- `MockBackend` for a deterministic demo when model infrastructure fails.

The phone communicates only with the companion API, never directly with vLLM.

## REST API contract

### Pairing and status

```text
GET  /v1/health
POST /v1/pair/start
POST /v1/pair/confirm
GET  /v1/capabilities
```

Example capabilities:

```json
{
  "server_name": "Sinan's Mac",
  "base_model": "gemma-4",
  "inference_ready": true,
  "training_available": true,
  "supported_file_types": ["pdf", "txt", "md"],
  "adapters": ["base", "personal-notes"]
}
```

### Conversations

```text
POST /v1/chat/completions
GET  /v1/chat/completions/{request_id}/events
```

Request:

```json
{
  "conversation_id": "uuid",
  "message": "What were the three decisions in my project notes?",
  "workspace_id": "uuid",
  "adapter_id": "personal-notes",
  "stream": true
}
```

The response stream emits:

- `route`;
- `retrieval`;
- `token`;
- `citation`;
- `complete`;
- `error`.

### Documents

```text
POST   /v1/workspaces
POST   /v1/workspaces/{workspace_id}/documents
GET    /v1/workspaces/{workspace_id}/documents
DELETE /v1/workspaces/{workspace_id}/documents/{document_id}
```

### Training and adapters

```text
POST   /v1/workspaces/{workspace_id}/training-jobs
GET    /v1/training-jobs/{job_id}
GET    /v1/training-jobs/{job_id}/events
GET    /v1/adapters
DELETE /v1/adapters/{adapter_id}
```

The public companion API owns adapter lifecycle. Raw vLLM dynamic adapter-management
routes remain private to localhost.

## Request orchestration

The server builds a bounded context in this order:

1. system and privacy policy;
2. stable adapter/task instructions;
3. top retrieved chunks with source metadata;
4. compact conversation summary;
5. recent conversation turns;
6. current user question.

It records:

- input token estimate;
- retrieved chunk identifiers;
- selected adapter;
- time to first token;
- total latency;
- response citations.

This telemetry stays local and becomes useful evidence for the hackathon pitch.

## Security and privacy rules

- Default to on-device inference.
- Require an explicit user action before uploading documents.
- Bind the companion service to the trusted local network only.
- Pair devices and authenticate every non-pairing request.
- Keep vLLM's runtime LoRA management endpoint on localhost.
- Reject arbitrary adapter filesystem paths from phone requests.
- Use generated workspace and adapter identifiers.
- Validate file type, size, and extracted text length.
- Provide delete controls for documents, indexes, adapters, and conversations.
- Do not claim end-to-end encryption until transport encryption is implemented and
  verified.

## Hackathon MVP

### Must work

1. Existing Gemma chat continues to run on iPhone.
2. A microphone button produces a local transcript.
3. The transcript can be routed to a configured companion-server URL.
4. The server sends it to Gemma and streams the response back.
5. The phone displays and speaks the response.
6. The route is visible: `On this iPhone` or `On paired Mac`.
7. One text or PDF document can be uploaded and indexed.
8. A question retrieves cited context from that document.
9. A training job has visible status and produces or loads one demonstrable adapter.

### Can be simulated safely

- Bonjour discovery can be replaced by entering the Mac URL or scanning a QR code.
- Pairing can use a pre-shared demo token.
- A pre-trained adapter can be used while the UI shows a short reproducible training
  job.
- PDF support can fall back to plain text if extraction becomes a time sink.

### Explicitly out of scope

- background always-listening audio;
- arbitrary remote internet access;
- multi-user accounts;
- production-grade certificate infrastructure;
- training the base model;
- syncing raw personal data to a third-party cloud;
- supporting every document format.

## Implementation sequence

### Phase 1 — Voice loop

1. Copy or continue from the existing `cactus` app.
2. Add microphone and speech usage descriptions to `Info.plist`.
3. Implement `SpeechInputService`.
4. Add a microphone button and live transcript state.
5. Implement `SpeechOutputService`.
6. Speak the existing on-device Gemma response.

Acceptance test: airplane mode on; speak a question; transcript, answer, and spoken
output all work on the iPhone.

### Phase 2 — Hybrid inference

1. Create the FastAPI companion skeleton.
2. Add health, capabilities, and chat endpoints.
3. Add a vLLM-compatible backend client.
4. Add `ComputeNodeClient` to the iPhone.
5. Implement routing and a visible route badge.
6. Stream tokens or sentences back to the app.

Acceptance test: turn the companion server on and off; the route changes visibly and
the app falls back to local inference.

### Phase 3 — Documents and retrieval

1. Add the iOS document picker.
2. Upload one supported file with progress.
3. Extract, chunk, index, and persist it on the node.
4. Retrieve top chunks for a question.
5. Include sources in the answer and phone UI.

Acceptance test: ask one question whose answer exists only in the uploaded document
and show the cited chunk.

### Phase 4 — Personal adapter

1. Convert one workspace into a small instruction dataset.
2. Train or provide a pre-trained LoRA.
3. Record base-model compatibility and evaluation output.
4. Register the adapter through the internal backend.
5. Select the adapter from the phone.
6. Compare base Gemma and personalized Gemma on one stable prompt.

Acceptance test: the same prompt produces a visibly intended change with the adapter,
while retrieved factual answers remain cited.

### Phase 5 — Submission

1. Record the offline voice flow.
2. Record Mac handoff and route switching.
3. Record document upload and cited answer.
4. Show adapter training status and before/after output.
5. Publish setup instructions and architecture.
6. Write the Kaggle submission around privacy, graceful scaling, and user ownership.

## Demo script

1. Put the phone in airplane mode.
2. Ask by voice: “Give me three ideas for tonight’s demo.”
3. Show local transcription, Gemma on-device inference, and local spoken output.
4. Re-enable the local network and connect to the Mac.
5. Upload a short private project document.
6. Ask a question that only that document can answer.
7. Show the answer, citation, and `Paired Mac` route badge.
8. Switch from base Gemma to the personal adapter.
9. Repeat a style/task prompt and show the intended difference.
10. Disconnect the Mac and show that the phone still works.

## Success metrics

- voice-to-transcript success on the demo phrase;
- successful offline completion;
- successful paired-node completion;
- visible fallback when the node disappears;
- time to first token for phone versus node;
- retrieved source shown for document questions;
- reduced prompt size from selecting only relevant document chunks;
- measurable base-versus-adapter change on a fixed evaluation set;
- zero calls to a third-party AI API during the demo.

## Main technical risks and mitigations

### vLLM on the available Mac

vLLM-Metal can provide the OpenAI-compatible vLLM path on Apple Silicon, but Gemma 4
support is experimental and LoRA support is not yet a dependable Metal feature. Keep
the companion API independent from the backend. Use base Gemma through vLLM-Metal for
the Mac demo; use NVIDIA/Linux vLLM for dynamic adapters, or merge an MLX-trained
adapter into a separately served Mac checkpoint.

### LoRA training time

Training during a short hackathon can fail or take too long. Prepare a known-good
adapter and a tiny deterministic training path. Never make the entire voice demo
depend on live training.

### Document facts learned by LoRA

Fine-tuning can distort or omit facts. Use retrieval for factual grounding and LoRA
for behavior. Always display citations for document answers.

### Local network setup

Venue Wi-Fi may block peer discovery. Bring a phone hotspot or USB/network fallback,
and allow manual server URL entry or QR-code configuration.

### Model compatibility

An adapter is tied to a specific base model and module layout. Pin model identifiers,
revisions, tokenizer, training configuration, and adapter metadata.

## Suggested repository layout

```text
hackathon_gemma/
├── ios/
│   ├── GemmaMesh.xcodeproj
│   └── GemmaMesh/
│       ├── App/
│       ├── Chat/
│       ├── Speech/
│       ├── ComputeNode/
│       ├── Documents/
│       └── Storage/
├── server/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── inference/
│   │   ├── retrieval/
│   │   ├── training/
│   │   └── storage/
│   ├── tests/
│   └── pyproject.toml
├── demo/
│   ├── sample_document.md
│   └── evaluation_prompts.json
├── PROJECT_SPEC.md
└── README.md
```

## References

- Apple Speech framework:
  <https://developer.apple.com/documentation/speech/>
- Apple `AVSpeechSynthesizer`:
  <https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer>
- vLLM LoRA adapters:
  <https://docs.vllm.ai/en/latest/features/lora/>
- vLLM security guidance:
  <https://docs.vllm.ai/en/latest/usage/security/>
- Hugging Face PEFT:
  <https://huggingface.co/docs/transformers/en/peft>

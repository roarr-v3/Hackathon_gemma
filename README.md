# Gemma Personal Compute Mesh

A local-first iPhone assistant that runs Gemma 4 E2B on-device with Cactus and
automatically moves document-backed questions to a trusted Mac compute node.

## Project video
[Watch the Gemma Personal Compute Mesh demo on YouTube](https://youtube.com/shorts/b-Wt3Tg_DCM)

## Platform target

This hackathon build intentionally targets **iPhone on iOS 27 only** and assumes
**macOS 27 with Xcode 27** for development and the Mac compute node. It is optimized
for recent Pro-class hardware such as iPhone 16 Pro and iPhone 17 Pro. Backward
compatibility, iPad, and older-device memory compromises are not product goals.

## What is implemented

- The working Cactus iOS app is ported under `ios/`.
- Prompts without documents continue to use local Cactus inference.
- The paperclip opens the iOS document picker.
- Selecting a document immediately uploads it to the configured compute node.
- While any document is attached, the app requires the compute node for inference.
- The app never silently falls back to a document-free local answer.
- The compute-node tab stores the server URL and API token and tests `/v1/health`.
- The companion service accepts text, Markdown, JSON, CSV, and PDF files.
- The companion service chunks documents, selects relevant chunks, and supplies
  cited context to Gemma through an OpenAI-compatible vLLM endpoint.
- Server responses include the selected model and document citations.
- The microphone produces a live, local transcript using the iOS 27
  `SpeechTranscriber`, `SpeechAnalyzer`, and `AnalyzerInputConverter` APIs.
- Completed Gemma responses are spoken locally with the best installed system
  voice; the speaker button can disable or interrupt spoken replies.

The companion already accepts an optional `adapter_id` and forwards it as the
vLLM model name. Experimental runtime LoRA registration is verified with the
installed vLLM-Metal development build; document-to-LoRA generation still needs
an E4B-trained hypernetwork.

## System boundary

```text
iPhone
  Microphone ──> on-device STT ──> transcript
  ├─ no document ────────> Cactus + Gemma 4 E2B on iPhone
  └─ document attached ──> FastAPI companion on Mac
                               ├─ private document store/retrieval
                               └─ OpenAI-compatible API
                                      └─ Gemma 4 E4B on vLLM-Metal
  Answer ──────> on-device system TTS
```

The iPhone talks to the companion API on port `8080`. The companion talks to vLLM
on port `8000`. Do not expose vLLM's adapter-management endpoints to the iPhone.

## Voice loop

On the iOS 27 demo phone:

1. Tap the microphone and grant microphone and Speech Recognition permission.
2. The first use may briefly prepare or download Apple's locale-specific speech
   asset.
3. Speak while the live transcript appears in the message field.
4. Tap the red stop button, correct the transcript if needed, and send it.
5. The response is displayed and spoken locally.

Starting the microphone stops any current spoken answer. The app requests local
speech processing and never deliberately falls back to network recognition. The
speaker button in the navigation bar toggles automatic spoken replies.

## 1. Start the inference engine

The Cactus model bundle is not directly reusable by vLLM. Cactus uses its own
quantized bundle format; the compute node needs the Hugging Face Gemma checkpoint
supported by its vLLM backend.

### Apple Silicon Mac: vLLM-Metal

Current vLLM documentation provides a separate vLLM-Metal plugin for Apple Silicon.
It requires native arm64 Python 3.12. Gemma 4 support is currently marked
experimental, so keep mock mode available for the demo.

Install vLLM-Metal using its official instructions, then:

```bash
source ~/.venv-vllm-metal/bin/activate
export HF_TOKEN="your-hugging-face-token-without-angle-brackets"
vllm serve mlx-community/gemma-4-e4b-it-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096
```

Use `127.0.0.1`, not `0.0.0.0`, when vLLM and the companion run on the same Mac.
Only the authenticated companion service needs to be visible to the iPhone.
Do not include literal `<` or `>` characters around the Hugging Face token. Once the
model is cached locally, vLLM can normally start without downloading it again.

The `4096` context default is intentional for memory-constrained Macs. Increase it
only after the server starts reliably; vLLM prints the estimated maximum context
length when KV-cache memory is insufficient.

The compute node intentionally uses E4B while the memory-constrained iPhone keeps
using E2B. On Apple Silicon, use the MLX conversion above. The upstream
`google/gemma-4-E4B-it-qat-mobile-ct` checkpoint uses compressed-tensors packed
weights; the current vLLM-Metal MLX loader does not load those weights directly.
That upstream checkpoint remains appropriate for a compatible CUDA/mainline-vLLM
node.

### Experimental runtime LoRA on E4B

The locally installed vLLM-Metal development build can wrap E4B's quantized
`down_proj` layers, dynamically register a PEFT adapter, and select it per request.
Start it with:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_METAL_USE_PAGED_ATTENTION=1
vllm serve mlx-community/gemma-4-e4b-it-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 8 \
  --lora-target-modules down_proj
```

The runtime update endpoint must remain bound to localhost. Do not expose it to
the phone or the LAN.

Create the zero-effect adapter used to test E4B tensor shapes and routing:

```bash
~/.venv-vllm-metal/bin/python \
  server/scripts/create_e4b_test_adapter.py \
  server/data/adapters/e4b-zero-test
```

Register it while vLLM is running:

```bash
curl http://127.0.0.1:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "e4b-zero-test",
    "lora_path": "'"$PWD"'/server/data/adapters/e4b-zero-test"
  }'
```

Then use `"adapter_id": "e4b-zero-test"` in a companion chat request. The test
adapter deliberately contains zeros, so it proves dynamic loading and routing
without changing model behavior.

The current vLLM-Metal development build also enables Gemma 4's YOCO fast-prefill
optimization by default. Its reduced query rows are not yet synchronized with
LoRA's full token routing map. On this Mac the installed package has a local
compatibility patch that disables YOCO fast prefill whenever LoRA is enabled.
Reinstalling vLLM-Metal will overwrite that patch.

Sakana AI's checked-in Doc-to-LoRA checkpoint cannot directly produce an E4B
adapter. It is trained and hard-coded for `google/gemma-2-2b-it`, while E4B has a
different layer count and projection dimensions. Its architecture and training
pipeline are still useful, but the hypernetwork output heads must be retrained
for E4B before their generated tensors can be registered through the working
runtime path above.

If Gemma 4 E2B cannot load through vLLM-Metal on the available machine, MLX-LM also
provides an OpenAI-style `/v1/chat/completions` server. The companion code only
depends on that API contract, so it can use either backend:

```bash
mlx_lm.server --model <compatible-gemma-4-model>
```

Set `VLLM_BASE_URL` to the MLX server's `/v1` URL and set `VLLM_MODEL` to the exact
model name accepted by that server.

### Optional: use the installed Gemma 4 12B in LM Studio

This Mac already has a 6.7 GB `gemma-4-12B-it-QAT-Q4_0.gguf` checkpoint. It is a
GGUF model for LM Studio's llama.cpp runtime, not an MLX model and not a model that
vLLM-Metal can load directly. The companion accepts any OpenAI-compatible endpoint,
so it can still use this model as an optional higher-quality route:

```bash
lms server start --port 1234 --bind 127.0.0.1
lms load gemma-4-12B-it-QAT-Q4_0.gguf \
  --identifier gemma-4-12b \
  --context-length 4096 \
  --gpu max

export VLLM_BASE_URL="http://127.0.0.1:1234/v1"
export VLLM_MODEL="gemma-4-12b"
```

Despite the environment variable's historical `VLLM_` name, the companion only
requires the OpenAI chat-completions contract. E4B MLX is the recommended demo
default because it leaves substantially more unified memory available for Xcode,
retrieval, and the companion process.

### NVIDIA/Linux compute node

For a supported Linux GPU machine:

```bash
export HF_TOKEN="your-hugging-face-token-without-angle-brackets"
vllm serve google/gemma-4-E4B-it-qat-mobile-ct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --enable-lora
```

Accept the Gemma license on Hugging Face before the first model download.

If vLLM runs on a different machine from the companion, bind it to the private
network, configure an API key and firewall, and point `VLLM_BASE_URL` at that
machine.

### Test without loading Gemma

Mock mode validates document transfer, routing, authentication, and the entire app
flow:

```bash
export GEMMA_MOCK_MODE=true
```

The response clearly identifies itself as a mock and should only be used while
testing connectivity.

## 2. Start the companion API

From the repository root:

```bash
cd server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the environment:

```bash
export GEMMA_SERVER_NAME="Sinan's Mac"
export GEMMA_API_TOKEN="replace-this-demo-token"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_MODEL="mlx-community/gemma-4-e4b-it-4bit"
```

Start the API on the Mac's local-network interface:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Test it on the Mac:

```bash
curl http://127.0.0.1:8080/v1/health \
  -H "X-API-Key: replace-this-demo-token"
```

Expected shape:

```json
{
  "status": "ok",
  "server_name": "Sinan's Mac",
  "inference_ready": true,
  "model": "mlx-community/gemma-4-e4b-it-4bit"
}
```

Find the Mac's Wi-Fi address:

```bash
ipconfig getifaddr en0
```

If that prints, for example, `192.168.1.42`, the iPhone URL is:

```text
http://192.168.1.42:8080
```

The Mac and iPhone must be on the same network. Venue Wi-Fi may isolate clients, so
a phone hotspot is a safer hackathon setup.

## 3. Configure the iPhone

1. Open `ios/CactusApp.xcodeproj` in full Xcode.
2. Select the existing development team and a physical iPhone.
3. Build and run the app.
4. Accept the local-network permission prompt.
5. Open the **Compute** tab.
6. Enter `http://<MAC-IP>:8080`.
7. Enter the same `GEMMA_API_TOKEN`.
8. Tap **Save and test**.
9. Return to Chat and attach a small `.txt`, `.md`, or `.pdf` file.
10. Wait for the green checkmark, ask a question, and send.

`localhost` on a physical iPhone is the iPhone itself, not the Mac.

The current hackathon build permits plain HTTP so a private LAN demo works without
certificate setup. This is intentionally not a production transport configuration.
Production should use TLS plus real pairing credentials.

## REST flow

### Health

```http
GET /v1/health
X-API-Key: <token>
```

### Upload

```http
POST /v1/documents
X-API-Key: <token>
Content-Type: multipart/form-data
```

The response contains the server-side document ID:

```json
{
  "id": "uuid",
  "name": "notes.md",
  "character_count": 4200
}
```

### Document-backed chat

```http
POST /v1/chat/completions
X-API-Key: <token>
Content-Type: application/json
```

```json
{
  "conversation_id": "uuid",
  "message": "What deadline is mentioned in my notes?",
  "document_ids": ["document-uuid"],
  "adapter_id": null
}
```

The companion retrieves relevant chunks, builds the bounded Gemma context, calls
vLLM, and returns:

```json
{
  "answer": "The deadline is Friday. [Source 1]",
  "model": "mlx-community/gemma-4-e4b-it-4bit",
  "citations": [
    {
      "document_id": "document-uuid",
      "document_name": "notes.md",
      "chunk_id": "document-uuid:0"
    }
  ]
}
```

## How Swift is connected

`ComputeNodeClient.swift` performs three jobs:

1. `health()` calls the companion API and verifies the node.
2. `uploadDocument(at:)` reads the security-scoped file and sends multipart data.
3. `generate(...)` sends the prompt and uploaded document IDs as JSON.

`ChatViewModel.send` is the router:

```swift
if requiresComputeNode {
    // POST to the companion API with document IDs.
} else {
    // Existing CactusInferenceService on the iPhone.
}
```

That single condition is the product rule. A failed upload or unreachable node stops
the request and shows an error; it does not quietly invoke the phone model without
the requested document.

## Validation

Checks that do not require downloading model weights:

```bash
python3 -m compileall -q server/app server/tests
plutil -lint ios/CactusApp/Info.plist \
  ios/CactusApp.xcodeproj/project.pbxproj
/Applications/Xcode-beta.app/Contents/Developer/usr/bin/xcodebuild \
  -project ios/CactusApp.xcodeproj \
  -scheme CactusApp \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  ARCHS=arm64 ONLY_ACTIVE_ARCH=YES CODE_SIGNING_ALLOWED=NO build
```

After installing the server dependencies:

```bash
cd server
PYTHONPATH=. .venv/bin/pytest -q
```

## Next phases

1. Stream response tokens with server-sent events.
2. Add a workspace and document deletion UI.
3. Generate a deterministic training dataset from a workspace.
4. Train a PEFT/MLX LoRA tied to the exact Gemma base revision.
5. On NVIDIA/Linux, register approved adapters with vLLM. On Apple Silicon, merge
   the selected adapter into a served checkpoint until vLLM-Metal supports LoRA.
6. Select an adapter through a safe `adapter_id`, never a client-provided path.
7. Add optional Foundation Models routing and App Intents only after the core demo
   is stable on the physical phone.

See `PROJECT_SPEC.md` for the full architecture, security model, demo, and training
plan.

## Primary references

- vLLM Gemma 4 recipe:
  <https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md>
- vLLM-Metal installation:
  <https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/>
- vLLM-Metal support matrix:
  <https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md>
- vLLM LoRA serving:
  <https://docs.vllm.ai/en/latest/features/lora/>
- MLX-LM local server:
  <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md>

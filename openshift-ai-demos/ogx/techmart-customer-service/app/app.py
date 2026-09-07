"""
TechMart Customer Service Assistant - Web UI
Integrates with OGX (RAG + MCP) for intelligent customer support
"""

from flask import Flask, render_template, request, jsonify, session
from ogx_client import OgxClient
import os
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'techmart-demo-secret-key-change-in-production')

# Limit uploaded files to 5 MB to prevent blocking the server on large uploads.
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Configuration
OGX_URL = os.environ.get('OGX_URL', 'http://localhost:8321')
MCP_SERVER_URL = os.environ.get('MCP_SERVER_URL', 'http://localhost:9001/sse')

# Universal instruction for all scenarios
INSTRUCTIONS = """You are a TechMart customer service assistant. Today is April 21, 2024. Be friendly, natural, and concise, and answer only what the question asks. Never show your reasoning.

TOOLS — pick the right one, and never do the math yourself:
- Return eligibility / refund ("Can I return ORD-...?", "Am I eligible?", "What's my refund?"): you MUST call check_return_eligibility and base your answer entirely on it. It returns eligible, deadline, days_remaining, return_window_days, restocking_fee_percent, restocking_fee_amount, and refund_amount already computed. Use those exact values — never decide eligibility yourself and never recompute dates, fees, or refunds. Do NOT use get_order for these questions.
- Other order questions (status, tracking, delivery date, price, what was ordered): call get_order.
- Policy questions — anything about shipping times or costs, the return/refund policy, fees, warranty, or how to return an item: you MUST call file_search FIRST, then answer ONLY from the retrieved text, quoting its exact numbers. Never answer a policy question from memory, and do NOT say you don't have the information unless file_search actually returned nothing.
- Never invent days, fees, dates, or prices.

ANSWER FORMATS (fill the <values> from the tool output):
- Return eligible=true: "The order <id> is eligible for return. You have <days_remaining> days remaining in the return window, and the estimated refund amount is $<refund_amount>." If restocking_fee_percent > 0, add: " Additionally, there is a restocking fee of <restocking_fee_percent>% ($<restocking_fee_amount>) and you can return the item within <return_window_days> days of delivery."
- Return eligible=false: "Unfortunately, order <id> is not eligible for return. The return window expired <days_since_deadline> days ago, and you are outside the allowed timeframe."
- Order status/details: state the status, delivery date, and (if relevant) whether the package was opened, in 1-2 natural sentences.
- Specific policy question (one topic, e.g. "how long does shipping take?"): 1-2 sentences, keeping the policy's exact numbers.
- Broad policy question (e.g. "what is your return policy in general?"): reply with EXACTLY two sections and nothing else, using the retrieved text's numbers:
  "Return windows:" — one short bullet per product category.
  "Restocking fees:" — one short bullet per category, including the unopened case.
  Then one closing line offering more detail. Keep the per-category split: NEVER merge categories into a single number, and NEVER describe a category rule as an opened/unopened rule. Add NO other section — no refund calculations or examples, no return conditions, no shipping, no gift returns, no processing times, no last-updated date. Keep the whole answer under 120 words."""

def _item_attr(item, key):
    """Read an attribute from a Responses API output item (object or dict)."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _log_tool_calls(response) -> None:
    """Log the tool-call items the model produced, to detect misrouted calls.

    The Responses API returns an ``output`` list whose items include tool calls
    (e.g. ``file_search_call``, ``mcp_call``/``mcp_tool_call``,
    ``mcp_list_tools``). For MCP calls we also log the specific tool name (e.g.
    ``get_order`` vs ``check_return_eligibility``) so we can see, per question,
    exactly which tool the model routed to — the item type alone hides that.
    """
    output = getattr(response, "output", None)
    if not output:
        return
    calls = []
    for item in output:
        item_type = _item_attr(item, "type")
        if not item_type or item_type == "message":
            continue
        if item_type in ("mcp_call", "mcp_tool_call"):
            name = _item_attr(item, "name")
            calls.append(f"{item_type}:{name}" if name else item_type)
        else:
            calls.append(item_type)
    logger.info(f"Tool calls this turn: {calls or 'none'}")


def _strip_special_tokens(text: str) -> str:
    """Remove chat-template special tokens that can leak into the output.

    Some models emit delimiter tokens such as ``<|...|>`` (channel markers, stop
    tokens, or trailing IDs) that should never be shown to the customer. We strip
    both complete ``<|...|>`` tokens anywhere in the text and an unterminated
    ``<|...`` at the very end, which happens when generation is cut off
    (e.g. hitting max_output_tokens) mid-token.
    """
    text = re.sub(r"<\|[^|]*\|>", "", text)   # complete tokens anywhere
    text = re.sub(r"<\|[^|]*$", "", text)      # unterminated token at the end
    return text.strip()


# Helpers to handle model items that may be typed objects or raw tuples
def _model_type(m):
    if isinstance(m, tuple):
        return m[1] if len(m) > 1 else None
    return getattr(m, 'model_type', None)

def _model_id(m):
    if isinstance(m, tuple):
        return m[0]
    return getattr(m, 'identifier', None)

# Initialize OGX client
try:
    client = OgxClient(base_url=OGX_URL)
    logger.info(f"✅ Connected to OGX at {OGX_URL}")

    # Extract LLM model ID dynamically
    # .list() may return a SyncPage (.data) or a plain iterable of tuples/objects
    models_response = client.models.list()
    models = models_response.data if hasattr(models_response, 'data') else list(models_response)
    llm_model = next((m for m in models if _model_type(m) == "llm"), None)
    
    if not llm_model:
        logger.error("❌ No LLM model found in OGX")
        raise RuntimeError("No LLM model available")
    
    MODEL_ID = _model_id(llm_model)
    logger.info(f"✅ Using LLM model: {MODEL_ID}")
        
except Exception as e:
    logger.error(f"❌ Failed to initialize OGX: {e}")
    client = None
    MODEL_ID = None

# Last-resolved vector store ID, reported by /api/status. The store itself is
# looked up by name on every use, so this is a reporting value, not a cache.
VECTOR_STORE_ID = None

def get_or_create_vector_store():
    """Get or create vector store for RAG"""
    global VECTOR_STORE_ID

    if not client:
        return None

    try:
        # Always resolve by name rather than trusting VECTOR_STORE_ID. The app
        # runs under multiple gunicorn workers, each with its own copy of this
        # global: when one worker replaces the store on upload, the others would
        # otherwise keep using an ID that no longer exists. The lookup is a
        # single in-cluster request, so re-resolving is cheap.
        vector_stores = client.vector_stores.list()
        for vs in vector_stores:
            if vs.name == "techmart_policy_store":
                VECTOR_STORE_ID = vs.id
                # Debug, not info: this now runs on every lookup, and the UI
                # polls /api/files.
                logger.debug(f"Using existing vector store: {VECTOR_STORE_ID}")
                return VECTOR_STORE_ID
        
        # Get models to find embedding model
        models_response = client.models.list()
        models = models_response.data if hasattr(models_response, 'data') else list(models_response)
        embedding_model = next((m for m in models if _model_type(m) == "embedding"), None)

        if not embedding_model:
            logger.error("No embedding model found")
            return None

        embedding_model_id = _model_id(embedding_model)
        embedding_dimension = int(
            embedding_model.metadata.get("embedding_dimension", 384)
            if hasattr(embedding_model, 'metadata')
            else 384
        )
        
        # Create new vector store
        vector_store = client.vector_stores.create(
            name="techmart_policy_store",
            extra_body={
                "embedding_model": embedding_model_id,
                "embedding_dimension": embedding_dimension,
                "provider_id": "faiss",
            },
        )
        
        VECTOR_STORE_ID = vector_store.id
        logger.info(f"✅ Created vector store: {VECTOR_STORE_ID}")
        return VECTOR_STORE_ID
        
    except Exception as e:
        logger.error(f"❌ Error with vector store: {e}")
        return None

def reset_vector_store():
    """Drop already-indexed content so the next upload replaces it instead of adding to it.

    Re-uploading an edited policy must not leave the previous version's chunks
    behind — retrieval would mix old and new numbers. Deleting the store is how
    we guarantee that, but it is only worth doing when the store actually holds
    files. On a new instance the store is empty (or does not exist yet), so
    there is nothing to replace: keep it and let the upload populate it.
    Deleting and recreating an empty store just churns IDs for no benefit.
    """
    global VECTOR_STORE_ID
    if not client:
        return
    try:
        for vs in client.vector_stores.list():
            if vs.name != "techmart_policy_store":
                continue
            if not any(True for _ in client.vector_stores.files.list(vector_store_id=vs.id)):
                logger.info(f"↩️  Vector store is empty, reusing it for this upload: {vs.id}")
                VECTOR_STORE_ID = vs.id
                return
            client.vector_stores.delete(vector_store_id=vs.id)
            logger.info(f"🗑️  Deleted existing vector store for clean re-index: {vs.id}")
            VECTOR_STORE_ID = None
            return
        VECTOR_STORE_ID = None
    except Exception as e:
        logger.warning(f"Could not reset existing vector store (continuing): {e}")

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if not client:
            return jsonify({'error': 'OGX client not initialized'}), 500
        
        logger.info(f"Processing message: {user_message[:50]}...")
        
        # Build tools list
        tools = []
        
        # Add file_search tool if vector store exists
        vector_store_id = get_or_create_vector_store()
        if vector_store_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            })
        
        # Add MCP tool
        tools.append({
            "type": "mcp",
            "server_label": "TechMartOrdersServer",
            "server_url": MCP_SERVER_URL,
        })
        
        # Create response using OGX.
        # Each question needs one tool call (check_return_eligibility, get_order,
        # or file_search); the cap leaves a little headroom while avoiding wasted
        # model passes on CPU. Cap output tokens so answers stay short —
        # generation length is the main latency driver on the CPU runtime.
        response = client.with_options(timeout=600.0).responses.create(
            model=MODEL_ID,
            input=user_message,
            stream=False,
            max_tool_calls=3,
            max_output_tokens=250,
            instructions=INSTRUCTIONS,
            tools=tools,
        )
        
        # Log which tools the model actually invoked, so we can spot unnecessary
        # RAG (file_search) calls on questions that don't need policy lookup.
        _log_tool_calls(response)

        # Surface truncation explicitly. When generation hits max_output_tokens
        # the answer simply stops mid-sentence, which is easy to mistake for a
        # bad answer rather than a budget that is too small for the prompt.
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
            logger.warning(f"⚠️  Response truncated before completion (reason: {reason})")

        # Extract response text
        bot_response = response.output_text if hasattr(response, 'output_text') else str(response)

        # Strip any chat-template special tokens (e.g. <|...|>) that can leak
        # into the model output before showing it to the customer.
        bot_response = _strip_special_tokens(bot_response)

        # If the model produced no usable text (e.g. it exhausted its tool-call
        # budget or emitted only tokens we stripped), degrade to a readable
        # message instead of showing an empty chat bubble.
        if not bot_response:
            logger.warning("Empty model response after processing; returning fallback message")
            bot_response = (
                "Sorry, I wasn't able to generate a response for that. "
                "Please try rephrasing your question."
            )

        return jsonify({
            'response': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.exception(f"❌ Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/rag', methods=['POST'])
def upload_rag_document():
    """Upload a document for RAG (vector store)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Uploading RAG document: {file.filename}")

        # Clear any previously indexed document so this upload fully replaces
        # it — no stale chunks left behind to retrieve. No-op when the store is
        # already empty.
        reset_vector_store()

        # Get or create vector store
        vector_store_id = get_or_create_vector_store()
        if not vector_store_id:
            return jsonify({'error': 'Failed to create vector store'}), 500
        
        # Upload file to OGX
        file_info = client.files.create(
            file=(file.filename, file.stream),
            purpose="assistants",  # API only accepts "assistants" or "batch"
        )
        
        logger.info(f"✅ Uploaded file: {file_info.id}")
        
        # Add file to vector store with chunking strategy
        vector_store_file = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_info.id,
            chunking_strategy={
                "type": "static",
                "static": {
                    # Larger chunks keep each policy section (e.g. the full
                    # Restocking Fees or Return Time Limits block) intact so the
                    # right numbers are retrieved together instead of split
                    # across chunks.
                    "max_chunk_size_tokens": 800,
                    "chunk_overlap_tokens": 150,
                },
            },
        )
        
        logger.info(f"✅ Added file to vector store: {file.filename}")
        
        return jsonify({
            'success': True,
            'message': f'Document "{file.filename}" uploaded successfully to RAG',
            'filename': file.filename,
            'file_id': file_info.id
        })
        
    except Exception as e:
        logger.exception(f"❌ Error uploading RAG document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check system status.

    The app itself never calls the MCP server — OGX invokes MCP tools in-cluster —
    so the app's reachability to MCP is not a meaningful health signal and is not
    reported here.
    """
    try:
        ogx_stack_status = "connected" if client else "disconnected"

        return jsonify({
            'ogx_stack': {
                'status': ogx_stack_status,
                'url': OGX_URL
            },
            'model_id': MODEL_ID,
            'vector_store_id': VECTOR_STORE_ID
        })
    except Exception as e:
        logger.error(f"❌ Error checking status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """List all uploaded files in the vector store"""
    try:
        vector_store_id = get_or_create_vector_store()
        if not vector_store_id:
            return jsonify({'files': []})
        
        # Get files from vector store
        vector_store_files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        
        files_list = []
        for vsf in vector_store_files:
            try:
                # VectorStoreFile object has 'id' not 'file_id'
                file_id = vsf.id if hasattr(vsf, 'id') else None
                if not file_id:
                    continue
                    
                # Get file details
                file_info = client.files.retrieve(file_id)
                files_list.append({
                    'id': file_info.id,
                    'filename': file_info.filename if hasattr(file_info, 'filename') else 'Unknown',
                    'size': file_info.bytes if hasattr(file_info, 'bytes') else 0,
                    'created_at': file_info.created_at if hasattr(file_info, 'created_at') else None,
                    'status': vsf.status if hasattr(vsf, 'status') else 'completed'
                })
            except Exception as e:
                logger.warning(f"Could not retrieve file: {e}")
                continue
        
        return jsonify({'files': files_list})
        
    except Exception as e:
        logger.error(f"❌ Error listing files: {e}")
        return jsonify({'files': []})

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear the current chat session"""
    try:
        if 'session_id' in session:
            old_session = session['session_id']
            session.pop('session_id')
            logger.info(f"Cleared session: {old_session}")
        
        return jsonify({
            'success': True,
            'message': 'Session cleared successfully'
        })
    except Exception as e:
        logger.error(f"❌ Error clearing session: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

# Made with Bob

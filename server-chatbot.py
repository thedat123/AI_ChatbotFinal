from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import time
import uvicorn
from typing import List, Dict, Any, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from queue import Queue
import uuid
from contextlib import asynccontextmanager

# Import your chatbot class
from RAG_LLM import MedicalSpecialistRAGChatbot

# Global variables for managing multiple chatbot instances
chatbot_pool = []
chatbot_lock = threading.Lock()
max_chatbot_instances = 5  # Adjust based on your server capacity
request_queue = asyncio.Queue()  # Use asyncio.Queue instead of regular Queue
active_requests = {}

# Initialize chatbot pool
async def initialize_chatbot_pool():
    global chatbot_pool
    print(f"Initializing {max_chatbot_instances} chatbot instances...")
    
    for i in range(max_chatbot_instances):
        try:
            chatbot = MedicalSpecialistRAGChatbot()
            # Check if initialize is async
            if asyncio.iscoroutinefunction(chatbot.initialize):
                success = await chatbot.initialize()
            else:
                success = chatbot.initialize()
                
            if success:
                chatbot_pool.append({
                    'instance': chatbot,
                    'busy': False,
                    'id': f"chatbot_{i}"
                })
                print(f"Chatbot instance {i} initialized successfully")
            else:
                print(f"Failed to initialize chatbot instance {i}")
        except Exception as e:
            print(f"Error initializing chatbot instance {i}: {str(e)}")
    
    if not chatbot_pool:
        raise Exception("Failed to initialize any chatbot instances")
    
    print(f"Successfully initialized {len(chatbot_pool)} chatbot instances")

# Get available chatbot instance
def get_available_chatbot():
    with chatbot_lock:
        for chatbot_info in chatbot_pool:
            if not chatbot_info['busy']:
                chatbot_info['busy'] = True
                return chatbot_info
    return None

# Release chatbot instance
def release_chatbot(chatbot_info):
    with chatbot_lock:
        chatbot_info['busy'] = False

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await initialize_chatbot_pool()
        # Start background task processor
        asyncio.create_task(process_requests())
    except Exception as e:
        print(f"Failed to initialize chatbot pool: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down chatbot instances...")

app = FastAPI(
    title="Multi-User Medical Specialist Virtual Assistant API",
    description="Concurrent API for Medical Specialist Virtual Assistant",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response models
class ChatRequest(BaseModel):
    query: str
    patient_id: str
    session_id: Optional[str] = None
    include_context: bool = False
    max_results: int = 5
    show_evaluation: bool = False
    user_id: Optional[str] = None  # New field for user identification

class ChatResponse(BaseModel):
    response: str
    session_id: str
    context: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Union[float, str, int]]] = None
    processing_time: float
    evaluation: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

class StreamingChatRequest(BaseModel):
    query: str
    patient_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class SessionRequest(BaseModel):
    patient_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
class SessionResponse(BaseModel):
    session_id: str
    is_new: bool

class QueueStatusResponse(BaseModel):
    queue_length: int
    estimated_wait_time: float
    your_position: Optional[int] = None

# Background task processor for handling requests concurrently
async def process_requests():
    """Background task to process queued requests"""
    while True:
        try:
            if not request_queue.empty():
                request_info = await request_queue.get()
                request_id = request_info['request_id']
                
                try:
                    # Process request directly in async context
                    result = await process_chat_request_async(request_info)
                    active_requests[request_id] = {
                        'status': 'completed',
                        'result': result,
                        'timestamp': time.time()
                    }
                except Exception as e:
                    active_requests[request_id] = {
                        'status': 'error',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                
                # Clean up old completed requests (older than 5 minutes)
                current_time = time.time()
                for req_id in list(active_requests.keys()):
                    if current_time - active_requests[req_id]['timestamp'] > 300:
                        del active_requests[req_id]
                        
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
        except Exception as e:
            print(f"Error in request processor: {str(e)}")
            await asyncio.sleep(1)

async def process_chat_request_async(request_info):
    """Process individual chat request asynchronously"""
    request_data = request_info['request']
    
    # Get available chatbot
    chatbot_info = get_available_chatbot()
    if not chatbot_info:
        raise Exception("No chatbot instances available")
    
    try:
        chatbot = chatbot_info['instance']
        start_time = time.time()
        
        # Handle session
        session_id = request_data.session_id
        if not session_id:
            # Check if get_active_session is async
            if asyncio.iscoroutinefunction(chatbot.history_manager.get_active_session):
                session_id = await chatbot.history_manager.get_active_session(request_data.patient_id)
            else:
                session_id = chatbot.history_manager.get_active_session(request_data.patient_id)
                
            if not session_id:
                # Check if create_session is async
                if asyncio.iscoroutinefunction(chatbot.history_manager.create_session):
                    session_id = await chatbot.history_manager.create_session(request_data.patient_id)
                else:
                    session_id = chatbot.history_manager.create_session(request_data.patient_id)
        
        # Store user message
        if asyncio.iscoroutinefunction(chatbot.history_manager.store_message):
            await chatbot.history_manager.store_message(
                session_id=session_id,
                is_from_patient=True,
                message_text=request_data.query
            )
        else:
            chatbot.history_manager.store_message(
                session_id=session_id,
                is_from_patient=True,
                message_text=request_data.query
            )
        
        # Get response from chatbot - this is the main async call
        if asyncio.iscoroutinefunction(chatbot.answer):
            response = await chatbot.answer(request_data.query)
        else:
            response = chatbot.answer(request_data.query)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get context if available
        retrieved_contexts = []
        if hasattr(chatbot, 'get_last_retrieved_contexts'):
            if asyncio.iscoroutinefunction(chatbot.get_last_retrieved_contexts):
                retrieved_contexts = await chatbot.get_last_retrieved_contexts()
            else:
                retrieved_contexts = chatbot.get_last_retrieved_contexts()
        
        # Store bot response
        query_type = "general"
        if hasattr(chatbot, 'determine_query_type'):
            if asyncio.iscoroutinefunction(chatbot.determine_query_type):
                query_type = await chatbot.determine_query_type(request_data.query)
            else:
                query_type = chatbot.determine_query_type(request_data.query)
        
        if asyncio.iscoroutinefunction(chatbot.history_manager.store_message):
            await chatbot.history_manager.store_message(
                session_id=session_id,
                is_from_patient=False,
                message_text=response,
                context=retrieved_contexts if retrieved_contexts else None,
                query_type=query_type,
                processing_time=processing_time
            )
        else:
            chatbot.history_manager.store_message(
                session_id=session_id,
                is_from_patient=False,
                message_text=response,
                context=retrieved_contexts if retrieved_contexts else None,
                query_type=query_type,
                processing_time=processing_time
            )
        
        # Create response
        chat_response = ChatResponse(
            response=response,
            session_id=session_id,
            processing_time=processing_time,
            request_id=request_info['request_id'],
            metrics={
                "processing_time": processing_time,
                "num_results": len(retrieved_contexts),
                "query_type": query_type,
                "chatbot_instance": chatbot_info['id']
            }
        )
        
        if request_data.include_context and retrieved_contexts:
            chat_response.context = retrieved_contexts
        
        return chat_response
        
    finally:
        # Always release the chatbot instance
        release_chatbot(chatbot_info)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active", 
        "service": "Multi-User Medical Specialist Virtual Assistant API",
        "chatbot_instances": len(chatbot_pool),
        "available_instances": len([c for c in chatbot_pool if not c['busy']])
    }

@app.get("/queue-status")
async def get_queue_status():
    """Get current queue status"""
    return QueueStatusResponse(
        queue_length=request_queue.qsize(),
        estimated_wait_time=request_queue.qsize() * 2.0  # Estimate 2 seconds per request
    )

@app.post("/session", response_model=SessionResponse)
async def create_or_get_session(request: SessionRequest):
    """Create a new session or get an existing one for a patient"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    # Use any available chatbot for session management
    chatbot_info = get_available_chatbot()
    if not chatbot_info:
        raise HTTPException(status_code=503, detail="All chatbot instances are busy. Please try again later.")
    
    try:
        chatbot = chatbot_info['instance']
        
        if request.session_id:
            if asyncio.iscoroutinefunction(chatbot.history_manager.get_session_info):
                session_info = await chatbot.history_manager.get_session_info(request.session_id)
            else:
                session_info = chatbot.history_manager.get_session_info(request.session_id)
                
            if session_info and session_info.get('patient_id') == request.patient_id:
                return SessionResponse(
                    session_id=request.session_id,
                    is_new=False
                )
            else:
                if asyncio.iscoroutinefunction(chatbot.history_manager.create_session):
                    session_id = await chatbot.history_manager.create_session(request.patient_id)
                else:
                    session_id = chatbot.history_manager.create_session(request.patient_id)
        else:
            if asyncio.iscoroutinefunction(chatbot.history_manager.get_active_session):
                session_id = await chatbot.history_manager.get_active_session(request.patient_id)
            else:
                session_id = chatbot.history_manager.get_active_session(request.patient_id)
        
        if not session_id:
            raise HTTPException(status_code=500, detail="Failed to create or get session")
        
        return SessionResponse(
            session_id=session_id,
            is_new=request.session_id is None or request.session_id != session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling session: {str(e)}")
    finally:
        release_chatbot(chatbot_info)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message - queued for concurrent processing"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Add to queue
    request_info = {
        'request_id': request_id,
        'request': request,
        'timestamp': time.time()
    }
    
    await request_queue.put(request_info)
    active_requests[request_id] = {
        'status': 'queued',
        'timestamp': time.time()
    }
    
    # Wait for processing (with timeout)
    timeout = 30  # 30 seconds timeout
    start_wait = time.time()
    
    while time.time() - start_wait < timeout:
        if request_id in active_requests:
            req_status = active_requests[request_id]
            
            if req_status['status'] == 'completed':
                result = req_status['result']
                del active_requests[request_id]  # Clean up
                return result
            elif req_status['status'] == 'error':
                error = req_status['error']
                del active_requests[request_id]  # Clean up
                raise HTTPException(status_code=500, detail=f"Error processing request: {error}")
        
        await asyncio.sleep(0.1)
    
    # Timeout
    if request_id in active_requests:
        del active_requests[request_id]
    raise HTTPException(status_code=408, detail="Request timeout")

@app.post("/chat/async")
async def chat_async(request: ChatRequest):
    """Submit chat request for async processing - returns request ID immediately"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    request_id = str(uuid.uuid4())
    
    request_info = {
        'request_id': request_id,
        'request': request,
        'timestamp': time.time()
    }
    
    await request_queue.put(request_info)
    active_requests[request_id] = {
        'status': 'queued',
        'timestamp': time.time()
    }
    
    return {
        "request_id": request_id,
        "status": "queued",
        "estimated_wait_time": request_queue.qsize() * 2.0
    }

@app.get("/chat/status/{request_id}")
async def get_chat_status(request_id: str):
    """Check status of async chat request"""
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    req_info = active_requests[request_id]
    
    if req_info['status'] == 'completed':
        result = req_info['result']
        del active_requests[request_id]  # Clean up
        return {"status": "completed", "result": result}
    elif req_info['status'] == 'error':
        error = req_info['error']
        del active_requests[request_id]  # Clean up
        return {"status": "error", "error": error}
    else:
        return {"status": req_info['status']}

@app.post("/chat/stream")
async def chat_stream(request: StreamingChatRequest):
    """Stream chat response in real-time"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    async def generate_response():
        chatbot_info = get_available_chatbot()
        if not chatbot_info:
            yield f"data: {json.dumps({'error': 'No chatbot instances available'})}\n\n"
            return
        
        try:
            chatbot = chatbot_info['instance']
            
            # Handle session
            session_id = request.session_id
            if not session_id:
                if asyncio.iscoroutinefunction(chatbot.history_manager.get_active_session):
                    session_id = await chatbot.history_manager.get_active_session(request.patient_id)
                else:
                    session_id = chatbot.history_manager.get_active_session(request.patient_id)
                    
                if not session_id:
                    if asyncio.iscoroutinefunction(chatbot.history_manager.create_session):
                        session_id = await chatbot.history_manager.create_session(request.patient_id)
                    else:
                        session_id = chatbot.history_manager.create_session(request.patient_id)
            
            # Store user message
            if asyncio.iscoroutinefunction(chatbot.history_manager.store_message):
                await chatbot.history_manager.store_message(
                    session_id=session_id,
                    is_from_patient=True,
                    message_text=request.query
                )
            else:
                chatbot.history_manager.store_message(
                    session_id=session_id,
                    is_from_patient=True,
                    message_text=request.query
                )
            
            yield f"data: {json.dumps({'status': 'processing', 'session_id': session_id})}\n\n"
            
            # Get response
            if asyncio.iscoroutinefunction(chatbot.answer):
                response = await chatbot.answer(request.query)
            else:
                response = chatbot.answer(request.query)
            
            # Store bot response
            if asyncio.iscoroutinefunction(chatbot.history_manager.store_message):
                await chatbot.history_manager.store_message(
                    session_id=session_id,
                    is_from_patient=False,
                    message_text=response
                )
            else:
                chatbot.history_manager.store_message(
                    session_id=session_id,
                    is_from_patient=False,
                    message_text=response
                )
            
            yield f"data: {json.dumps({'status': 'completed', 'response': response, 'session_id': session_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            release_chatbot(chatbot_info)
    
    return StreamingResponse(generate_response(), media_type="text/plain")

# Session management endpoints - updated to handle async methods
@app.get("/history/{session_id}")
async def get_session_history(session_id: str, limit: int = 20):
    """Get the message history for a specific session"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    chatbot_info = get_available_chatbot()
    if not chatbot_info:
        raise HTTPException(status_code=503, detail="All chatbot instances are busy")
    
    try:
        chatbot = chatbot_info['instance']
        if asyncio.iscoroutinefunction(chatbot.history_manager.get_session_history):
            history = await chatbot.history_manager.get_session_history(session_id, limit)
        else:
            history = chatbot.history_manager.get_session_history(session_id, limit)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")
    finally:
        release_chatbot(chatbot_info)

@app.get("/sessions/{patient_id}")
async def get_patient_sessions(patient_id: str):
    """Get all sessions for a specific patient"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    chatbot_info = get_available_chatbot()
    if not chatbot_info:
        raise HTTPException(status_code=503, detail="All chatbot instances are busy")
    
    try:
        chatbot = chatbot_info['instance']
        if asyncio.iscoroutinefunction(chatbot.history_manager.get_patient_sessions):
            sessions = await chatbot.history_manager.get_patient_sessions(patient_id)
        else:
            sessions = chatbot.history_manager.get_patient_sessions(patient_id)
        return {"patient_id": patient_id, "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patient sessions: {str(e)}")
    finally:
        release_chatbot(chatbot_info)

@app.put("/sessions/{session_id}/end")
async def end_session(session_id: str):
    """Mark a session as inactive"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    chatbot_info = get_available_chatbot()
    if not chatbot_info:
        raise HTTPException(status_code=503, detail="All chatbot instances are busy")
    
    try:
        chatbot = chatbot_info['instance']
        if asyncio.iscoroutinefunction(chatbot.history_manager.end_session):
            success = await chatbot.history_manager.end_session(session_id)
        else:
            success = chatbot.history_manager.end_session(session_id)
            
        if success:
            return {"status": "success", "message": "Session ended successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to end session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")
    finally:
        release_chatbot(chatbot_info)

@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, name: str):
    """Rename a session"""
    if not chatbot_pool:
        raise HTTPException(status_code=500, detail="No chatbot instances available")
    
    chatbot_info = get_available_chatbot()
    if not chatbot_info:
        raise HTTPException(status_code=503, detail="All chatbot instances are busy")
    
    try:
        chatbot = chatbot_info['instance']
        if asyncio.iscoroutinefunction(chatbot.history_manager.rename_session):
            success = await chatbot.history_manager.rename_session(session_id, name)
        else:
            success = chatbot.history_manager.rename_session(session_id, name)
            
        if success:
            return {"status": "success", "message": "Session renamed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to rename session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming session: {str(e)}")
    finally:
        release_chatbot(chatbot_info)

# For development
if __name__ == "__main__":
    uvicorn.run("server-chatbot:app", host="0.0.0.0", port=8000, reload=False)  # disable reload for better concurrency
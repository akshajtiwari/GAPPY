import os
import shutil
import datetime
import secrets
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, Base, engine
from . import models, schemas, crud, auth, ai
from .security import encrypt_data, decrypt_data
from .integrations import connectors as connectors_service
from . import workflows_service, pod_store
from . import settings_service, chat as chat_engine
from .integrations.web_search import web_search, WebSearchError
from .integrations import email_adapter


app = FastAPI(title="LifeOS API", version="1.0.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup():
    # Automatically create tables in local Postgres
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# -- Auth Routes --
@app.post("/api/auth/signup", response_model=schemas.UserResponse)
async def signup(user_in: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = await crud.get_user_by_email(db, email=user_in.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    hashed_pwd = auth.get_password_hash(user_in.password)
    user = await crud.create_user(db, user_in, hashed_pwd)
    return user

@app.post("/api/auth/login", response_model=schemas.Token)
async def login(user_in: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    user = await crud.get_user_by_email(db, email=user_in.email)
    if not user or not auth.verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# -- Item CRUD Routes --
@app.get("/api/items", response_model=List[schemas.ItemResponse])
async def list_items(
    type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.get_items(db, item_type=type)

@app.post("/api/items", response_model=schemas.ItemResponse)
async def create_item(
    item_in: schemas.ItemCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # Create the item in DB
    item = await crud.create_item(db, item_in)
    
    # If the item is a note, run AI insight analysis & auto-linking
    if item.type == "note":
        existing_notes = await crud.get_items(db, item_type="note")
        # Filter out the newly created note from comparisons
        existing_notes = [n for n in existing_notes if n.id != item.id]
        
        ai_res = await ai.analyze_note_and_suggest_links(item, existing_notes)
        
        # Save trace and connections in metadata
        item_meta = dict(item.metadata_json or {})
        item_meta["ai_analysis"] = {
            "trace": ai_res.get("trace"),
            "suggested_tasks": ai_res.get("suggested_tasks", []),
            "suggested_connections": ai_res.get("connections", [])
        }
        item.metadata_json = item_meta
        await db.flush()
        
        # Automatically create suggested tasks and connections if configured
        # (For MVP, we auto-create them directly so the command center works automatically!)
        for t in ai_res.get("suggested_tasks", []):
            task_due = None
            if t.get("due_date"):
                try:
                    task_due = datetime.datetime.strptime(t["due_date"], "%Y-%m-%d")
                except ValueError:
                    pass
            db_task = await crud.create_item(db, schemas.ItemCreate(
                type="task",
                title=t["title"],
                content=t.get("content", ""),
                priority=t.get("priority", "medium"),
                status="todo",
                due_date=task_due,
                metadata_json={"source_note_id": item.id, "auto_generated": True}
            ))
            # Link task to the note
            await crud.create_connection(db, schemas.ConnectionCreate(
                source_id=item.id,
                target_id=db_task.id,
                connection_type="suggested_task"
            ))
            
        # (We store connection suggestions in metadata_json so the user can accept them with a single click!)
        pass
                
    return item

@app.get("/api/items/{item_id}", response_model=schemas.ItemResponse)
async def get_item(
    item_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    item = await crud.get_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.put("/api/items/{item_id}", response_model=schemas.ItemResponse)
async def update_item(
    item_id: str,
    item_in: schemas.ItemUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    item = await crud.update_item(db, item_id, item_in)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.delete("/api/items/{item_id}")
async def delete_item(
    item_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    success = await crud.delete_item(db, item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"detail": "Item deleted"}

# -- Connections Routes --
@app.get("/api/connections", response_model=List[schemas.ConnectionDetailResponse])
async def list_connections(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.get_connections(db)

@app.post("/api/connections", response_model=schemas.ConnectionResponse)
async def create_connection(
    conn_in: schemas.ConnectionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.create_connection(db, conn_in)

@app.delete("/api/connections/{conn_id}")
async def delete_connection(
    conn_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    success = await crud.delete_connection(db, conn_id)
    if not success:
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"detail": "Connection deleted"}

@app.get("/api/items/{item_id}/connections", response_model=List[schemas.ConnectionDetailResponse])
async def get_item_connections(
    item_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.get_connections_by_item_id(db, item_id)

# -- Learning Routes --
@app.post("/api/learning/upload", response_model=schemas.ItemResponse)
async def upload_material(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # 1. Save locally
    filename = f"{int(datetime.datetime.utcnow().timestamp())}_{file.filename}"
    local_path = os.path.join(UPLOAD_DIR, filename)
    with open(local_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Extract text for the study-coach agent
    sdk_res = await ai.upload_learning_file_to_lemma(filename, local_path)
    if "error" in sdk_res:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not process file: {sdk_res['error']}"
        )

    # 3. Create Item representing this Study Material
    item_in = schemas.ItemCreate(
        type="study_material",
        title=file.filename,
        content=f"/learning/{filename}",
        status="todo",
        metadata_json={
            "local_path": local_path,
            "extracted_text": sdk_res.get("extracted_text", ""),
            "chars": sdk_res.get("chars", 0),
        }
    )
    return await crud.create_item(db, item_in)

@app.post("/api/learning/study")
async def start_study_session(
    material_id: str = Form(...),
    self_reported_confusion: str = Form(""),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # Fetch material item
    material = await crud.get_item(db, material_id)
    if not material or material.type != "study_material":
        raise HTTPException(status_code=404, detail="Study material not found")
        
    material_context = material.metadata_json.get("extracted_text", "")

    # Generate study plan & questions using the study-coach Lemma agent
    ai_res, search_results = await ai.generate_study_plan_and_questions(
        material_title=material.title,
        material_context=material_context,
        self_reported_confusion=self_reported_confusion
    )
    
    if "error" in ai_res:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Study plan generation failed: {ai_res['error']}"
        )
        
    # Store generated Weak Topics in DB
    created_topics = []
    for topic_data in ai_res.get("weak_topics", []):
        db_topic = await crud.create_item(db, schemas.ItemCreate(
            type="study_topic",
            title=topic_data["topic"],
            content=topic_data["reason"],
            status="weak",
            metadata_json={"source_material_id": material_id}
        ))
        created_topics.append(db_topic)
        # Link topic to source material
        await crud.create_connection(db, schemas.ConnectionCreate(
            source_id=material_id,
            target_id=db_topic.id,
            connection_type="weakness_of"
        ))
        
    # Store generated Revision Plan steps in DB
    created_revisions = []
    for plan_step in ai_res.get("revision_plan", []):
        due_date = None
        if plan_step.get("due_date"):
            try:
                due_date = datetime.datetime.strptime(plan_step["due_date"], "%Y-%m-%d")
            except ValueError:
                pass
        db_rev = await crud.create_item(db, schemas.ItemCreate(
            type="task",
            title=plan_step["title"],
            content=plan_step["content"],
            status="todo",
            priority=plan_step.get("priority", "medium"),
            due_date=due_date,
            metadata_json={"source_material_id": material_id, "is_study_revision": True}
        ))
        created_revisions.append(db_rev)
        # Link task to study material
        await crud.create_connection(db, schemas.ConnectionCreate(
            source_id=material_id,
            target_id=db_rev.id,
            connection_type="practice_of"
        ))
        
    # Save the practice questions in the study session response but don't commit them to DB items
    # (Or they can be answered on frontend dynamically)
    return {
        "practice_questions": ai_res.get("practice_questions", []),
        "search_results": search_results,
        "weak_topics_count": len(created_topics),
        "revision_steps_count": len(created_revisions)
    }

# -- NEW EXTENSION ENDPOINTS --

# 1. Life Ops
@app.get("/api/lifeops/weekly-review")
async def get_weekly_review(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    now = datetime.datetime.utcnow()
    # Fetch all tasks and deadlines
    tasks = await crud.get_items(db, limit=500, item_type="task")
    deadlines = await crud.get_items(db, limit=500, item_type="deadline")
    all_tasks = tasks + deadlines
    
    open_tasks = []
    slipped_tasks = []
    stale_followups = []
    
    three_days_ago = now - datetime.timedelta(days=3)
    
    for t in all_tasks:
        if t.status != "done":
            # Is it open task or slipped?
            if t.due_date and t.due_date < now:
                slipped_tasks.append(t)
            else:
                open_tasks.append(t)
                
            # Is it a follow-up?
            is_waiting = (t.status == "waiting" or (t.metadata_json and t.metadata_json.get("waiting_on")))
            if is_waiting and t.updated_at <= three_days_ago:
                stale_followups.append(t)
                
    ai_summary = await ai.generate_weekly_review_summary(open_tasks, slipped_tasks, stale_followups)
    return ai_summary

@app.post("/api/lifeops/commit-parse")
async def commit_parse(
    req: schemas.CommitmentInboxRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    return await ai.parse_commitment_inbox(req.text)

@app.get("/api/lifeops/stale-followups")
async def get_stale_followups_list(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    now = datetime.datetime.utcnow()
    three_days_ago = now - datetime.timedelta(days=3)
    
    tasks = await crud.get_items(db, limit=500, item_type="task")
    deadlines = await crud.get_items(db, limit=500, item_type="deadline")
    all_tasks = tasks + deadlines
    
    stale = []
    for t in all_tasks:
        if t.status != "done":
            is_waiting = (t.status == "waiting" or (t.metadata_json and t.metadata_json.get("waiting_on")))
            if is_waiting and t.updated_at <= three_days_ago:
                stale.append(t)
                
    return {
        "count": len(stale),
        "items": [
            {
                "id": t.id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "due_date": t.due_date,
                "metadata_json": t.metadata_json,
                "updated_at": t.updated_at
            } for t in stale
        ]
    }

# 2. Second Brain
@app.get("/api/brain/insights")
async def get_brain_insights(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    notes = await crud.get_items(db, limit=500, item_type="note")
    # Filter notes from last 30 days
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    recent_notes = [n for n in notes if n.created_at >= thirty_days_ago]
    insights = await ai.surface_brain_insights(recent_notes)
    return {"insights": insights}

@app.post("/api/brain/draft")
async def generate_draft(
    req: schemas.DraftGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    notes = []
    for nid in req.note_ids:
        note = await crud.get_item(db, nid)
        if note and note.type == "note":
            notes.append(note)
    if not notes:
        raise HTTPException(status_code=400, detail="No valid notes selected")
    
    draft_content = await ai.generate_draft_from_notes(notes, req.format)
    return {"draft": draft_content}

# 3. Learning Companion
@app.get("/api/learning/reviews/due", response_model=List[schemas.StudyReviewResponse])
async def get_due_reviews(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.get_due_study_reviews(db, user_id=current_user.id)

@app.post("/api/learning/reviews/submit")
async def submit_review(
    submission: schemas.StudyReviewSubmit,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    res = await crud.update_study_review(db, submission.review_id, submission.score)
    if not res:
        raise HTTPException(status_code=404, detail="Study review not found")
    return {"detail": "Review updated", "interval_days": res.interval_days, "due_date": res.due_date}

@app.post("/api/learning/test-submit")
async def submit_practice_test(
    submission: schemas.PracticeTestSubmit,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    ai_res = await ai.score_test_and_map_topics(submission.answers)
    
    created_topics = []
    for topic_data in ai_res.get("topic_strength", []):
        status = topic_data.get("status", "weak")
        db_topic = await crud.create_item(db, schemas.ItemCreate(
            type="study_topic",
            title=topic_data["topic"],
            content=f"Strength Score: {topic_data.get('score')}",
            status=status,
            metadata_json={"source_material_id": submission.material_id, "score": topic_data.get("score")}
        ))
        created_topics.append(db_topic)
        await crud.create_connection(db, schemas.ConnectionCreate(
            source_id=submission.material_id,
            target_id=db_topic.id,
            connection_type="weakness_of" if status == "weak" else "strength_of"
        ))
        
    created_revisions = []
    for plan_step in ai_res.get("suggested_revisions", []):
        due_date = None
        if plan_step.get("due_date"):
            try:
                due_date = datetime.datetime.strptime(plan_step["due_date"], "%Y-%m-%d")
            except ValueError:
                pass
        db_rev = await crud.create_item(db, schemas.ItemCreate(
            type="task",
            title=plan_step["title"],
            content=plan_step["content"],
            status="todo",
            priority=plan_step.get("priority", "medium"),
            due_date=due_date,
            metadata_json={"source_material_id": submission.material_id, "is_study_revision": True}
        ))
        created_revisions.append(db_rev)
        await crud.create_connection(db, schemas.ConnectionCreate(
            source_id=submission.material_id,
            target_id=db_rev.id,
            connection_type="practice_of"
        ))
        
    return {
        "topic_strength": ai_res.get("topic_strength", []),
        "suggested_revisions_count": len(created_revisions)
    }

@app.post("/api/learning/debrief")
async def pomodoro_debrief(
    req: schemas.PomodoroDebriefRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    ai_res = await ai.generate_study_debrief_insights(req.summary, req.confusion)
    
    debrief_note = await crud.create_item(db, schemas.ItemCreate(
        type="note",
        title=f"Study Session Debrief: {datetime.datetime.utcnow().strftime('%Y-%m-%d')}",
        content=f"Focus Summary: {req.summary}\nConfusion/Struggles: {req.confusion or 'None'}\n\nAI Insights:\n{ai_res.get('feedback', '')}\nSuggested Next Focus: {ai_res.get('suggested_next_focus', '')}",
        status="todo"
    ))
    
    for wt in ai_res.get("weak_topics", []):
        db_topic = await crud.create_item(db, schemas.ItemCreate(
            type="study_topic",
            title=wt["topic"],
            content=wt["reason"],
            status="weak",
            metadata_json={"source_debrief_id": debrief_note.id}
        ))
        await crud.create_connection(db, schemas.ConnectionCreate(
            source_id=debrief_note.id,
            target_id=db_topic.id,
            connection_type="weakness_of"
        ))
        
    return ai_res

@app.get("/api/learning/reviews/generate-quiz")
async def get_review_quiz(
    concept_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    concept = await crud.get_item(db, concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    questions = await ai.generate_spaced_repetition_quiz(concept.title, concept.content or "")
    return {"questions": questions}

# 4. Cross-Module
@app.get("/api/search", response_model=List[schemas.ItemResponse])
async def universal_search(
    q: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    items = await crud.get_items(db, limit=500)
    query_lower = q.lower()
    results = []
    for item in items:
        # Search title, content, status, priority, or tags
        title_match = query_lower in item.title.lower()
        content_match = (item.content and query_lower in item.content.lower())
        if title_match or content_match:
            results.append(item)
    return results

@app.get("/api/today")
async def get_today_view(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    now = datetime.datetime.utcnow()
    today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    today_end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
    
    items = await crud.get_items(db, limit=500)
    
    overdue = []
    due_today = []
    stale_followups = []
    
    three_days_ago = now - datetime.timedelta(days=3)
    
    for item in items:
        if item.type in ("task", "deadline") and item.status != "done":
            if item.due_date:
                if item.due_date < today_start:
                    overdue.append(item)
                elif today_start <= item.due_date <= today_end:
                    due_today.append(item)
            
            is_waiting = (item.status == "waiting" or (item.metadata_json and item.metadata_json.get("waiting_on")))
            if is_waiting and item.updated_at <= three_days_ago:
                stale_followups.append(item)
                
    due_reviews = await crud.get_due_study_reviews(db, user_id=current_user.id)
    
    # AI insights card from notes
    notes = [n for n in items if n.type == "note" and n.created_at >= (now - datetime.timedelta(days=30))]
    insights = []
    if notes:
        insights = await ai.surface_brain_insights(notes)
        
    insight_card = insights[0] if insights else {
        "title": "Welcome to LifeOS",
        "description": "Log your daily notes and tasks. The AI will automatically analyze your second brain and surface key connections here.",
        "action": "expand"
    }
        
    return {
        "overdue_tasks": overdue,
        "due_today_tasks": due_today,
        "stale_followups": {
            "count": len(stale_followups),
            "items": stale_followups
        },
        "due_reviews": {
            "count": len(due_reviews),
            "items": due_reviews
        },
        "insight_card": insight_card
    }

# -- Integrations Routes --

@app.get("/api/integrations")
async def list_integrations(
    current_user: models.User = Depends(auth.get_current_user)
):
    """Live Lemma connector catalog merged with real connection status."""
    return await connectors_service.list_integrations()

@app.get("/api/integrations/{name}/auth-url")
async def get_integration_auth_url(
    name: str,
    current_user: models.User = Depends(auth.get_current_user)
):
    try:
        auth_url = await connectors_service.get_connect_url(name)
        return {"auth_url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/integrations/{name}")
async def disconnect_integration(
    name: str,
    current_user: models.User = Depends(auth.get_current_user)
):
    ok = await connectors_service.disconnect(name)
    if not ok:
        raise HTTPException(status_code=404, detail="Integration not connected")
    return {"detail": f"Disconnected {name} successfully."}

@app.post("/api/integrations/{name}/test")
async def test_integration_connection(
    name: str,
    current_user: models.User = Depends(auth.get_current_user)
):
    connected = await connectors_service.is_connected(name)
    if not connected:
        return {"status": "unhealthy", "error_message": "Not connected yet."}
    return {"status": "healthy"}

# -- AI Assistant Quick-Add (Today page) --

@app.post("/api/assistant/query", response_model=schemas.AssistantQueryResponse)
async def assistant_query(
    req: schemas.AssistantQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Parse a natural-language line into a task via the commitment-parser agent and save it."""
    parsed = await ai.parse_commitment_inbox(req.query)
    due = None
    if parsed.get("due_date"):
        try:
            due = datetime.datetime.strptime(parsed["due_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    item = await crud.create_item(db, schemas.ItemCreate(
        type="task",
        title=parsed.get("title") or req.query,
        content=parsed.get("content", ""),
        priority=parsed.get("priority", "medium"),
        status="todo",
        due_date=due,
        metadata_json={"category": parsed.get("category"), "source": "assistant_quick_add"},
    ))
    return schemas.AssistantQueryResponse(
        intent="create_task",
        integration="lifeos",
        execution_status="success",
        response_message=f"Added task: {item.title}",
        suggested_actions=[{"type": "open_task", "id": str(item.id)}],
    )

# -- Workflows (agent → human approval → action) --

@app.get("/api/workflows")
async def list_workflows(current_user: models.User = Depends(auth.get_current_user)):
    return await workflows_service.list_workflows()

@app.post("/api/workflows/{name}/start")
async def start_workflow(
    name: str,
    req: schemas.WorkflowStartRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    inputs = dict(req.inputs or {})
    # commitment-intake writes a task owned by the app owner.
    if name == "commitment-intake":
        inputs.setdefault("user_id", pod_store.DEFAULT_OWNER)
        inputs.setdefault("today", datetime.date.today().strftime("%Y-%m-%d"))
    try:
        return await workflows_service.start_workflow(name, inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/workflows/approvals")
async def list_approvals(current_user: models.User = Depends(auth.get_current_user)):
    return await workflows_service.list_approvals()

@app.post("/api/workflows/decision")
async def workflow_decision(
    req: schemas.WorkflowDecisionRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    try:
        return await workflows_service.decide(req.run_id, req.node_id, req.approved)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -- Chat / Conversations --

@app.get("/api/chat/conversations", response_model=List[schemas.ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.list_conversations(db, current_user.id)

@app.post("/api/chat/conversations", response_model=schemas.ConversationResponse)
async def create_conversation(
    conv_in: schemas.ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return await crud.create_conversation(db, current_user.id, title=conv_in.title, tag=conv_in.tag)

@app.patch("/api/chat/conversations/{conv_id}", response_model=schemas.ConversationResponse)
async def update_conversation(
    conv_id: int,
    conv_in: schemas.ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    conv = await crud.update_conversation(db, conv_id, current_user.id,
                                          title=conv_in.title, tag=conv_in.tag)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@app.delete("/api/chat/conversations/{conv_id}")
async def delete_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    ok = await crud.delete_conversation(db, conv_id, current_user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"detail": "Conversation deleted"}

@app.get("/api/chat/conversations/{conv_id}/messages", response_model=List[schemas.MessageResponse])
async def get_conversation_messages(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    conv = await crud.get_conversation(db, conv_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return await crud.get_messages(db, conv_id)

@app.post("/api/chat/conversations/{conv_id}/send", response_model=schemas.ChatSendResponse)
async def send_chat_message(
    conv_id: int,
    req: schemas.ChatSendRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    conv = await crud.get_conversation(db, conv_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    settings = await settings_service.get_resolved_settings(db, current_user.id)
    user_row, assistant_row, tools_used = await chat_engine.send_message(
        db, current_user.id, conv, req.message, settings
    )
    return schemas.ChatSendResponse(
        conversation_id=conv.id,
        user_message=user_row,
        assistant_message=assistant_row,
        tools_used=tools_used
    )

# -- Settings --

@app.get("/api/settings", response_model=schemas.SettingsResponse)
async def get_settings(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    values = await settings_service.get_masked_settings(db, current_user.id)
    return schemas.SettingsResponse(values=values)

@app.put("/api/settings", response_model=schemas.SettingsResponse)
async def update_settings(
    req: schemas.SettingsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    values = await settings_service.update_settings(db, current_user.id, req.values)
    return schemas.SettingsResponse(values=values)

@app.post("/api/settings/test-email")
async def settings_test_email(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    settings = await settings_service.get_resolved_settings(db, current_user.id)
    ok = await email_adapter.test_email(settings)
    return {"status": "healthy" if ok else "unhealthy"}

# -- Web Search --

@app.post("/api/web-search", response_model=schemas.WebSearchResponse)
async def run_web_search(
    req: schemas.WebSearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    settings = await settings_service.get_resolved_settings(db, current_user.id)
    try:
        res = await web_search(req.query, settings, max_results=req.max_results or 6)
    except WebSearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web search failed: {e}")
    return schemas.WebSearchResponse(
        query=req.query,
        provider=res.get("provider", "unknown"),
        answer=res.get("answer"),
        results=[schemas.WebSearchResult(**r) for r in res.get("results", [])]
    )

# Mount static files at root
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")



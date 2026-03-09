import os
import glob
import uuid
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from google import genai
from google.genai import types
from supabase import create_client, Client
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ───────────────────────────────────────
# 환경변수
# ───────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY")

client_genai = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash-lite"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ───────────────────────────────────────
# 핵심 원칙 (CORE_PRINCIPLES)
# ───────────────────────────────────────
CORE_PRINCIPLES = """
╔══════════════════════════════════════════════════════════════╗
║       세특 AI 코치 — 모든 응답 전 반드시 준수할 핵심 원칙           ║
╚══════════════════════════════════════════════════════════════╝

[원칙 1] 세특은 반드시 3인칭 관찰 기록 문체로 작성한다.
  - "~함.", "~임.", "~을 보임." 형태의 명사형 종결어미를 사용한다.
  - "저는", "나는", "제가" 같은 1인칭 표현은 절대 금지.
  - 교사가 학생을 관찰하고 기록한 것처럼 자연스럽게 서술한다.

[원칙 2] 진로·희망학과와의 연결을 반드시 구체적으로 드러낸다.
  - 단순히 "진로와 연관됨"이 아니라 어떤 활동이 어떤 역량으로 이어지는지 명시한다.
  - 예: "경영학과 진학을 목표로 데이터 기반 의사결정 역량을 키우기 위해 ~을 탐구함."

[원칙 3] 구체적인 활동·사고 과정·성장이 드러나야 한다.
  - "열심히", "노력하는", "관심이 많은" 같은 추상적 표현은 절대 금지.
  - 무엇을 했는지, 어떻게 생각했는지, 어떤 결론을 냈는지가 보여야 한다.

[원칙 4] 주어진 분량을 반드시 지킨다.
  - 글자수 제한이 있으면 ±10자 이내로 맞춘다.
  - 바이트 제한이 있으면 한글 1자=3바이트, 영문/숫자 1자=1바이트로 계산한다.
  - 분량이 부족하면 구체적 사례나 사고 과정을 추가해 채운다.

[원칙 5] 불확실한 정보는 절대 포함하지 않는다.
  - 학생이 제공한 내용에 없는 활동·수상·성과를 지어내는 것은 절대 금지.
  - 애매한 내용은 일반적 표현으로 처리하되 없는 사실을 추가하지 않는다.

[원칙 6] 입시 관점에서 전략적으로 구성한다.
  - 대학 입학사정관이 주목하는 요소: 지적 호기심, 자기주도성, 진로 연계성, 성장 스토리.
  - 나열식이 아닌 하나의 스토리 흐름으로 구성한다.
  - 첫 문장은 학생의 특징적인 활동이나 질문으로 시작해 인상을 남긴다.
""".strip()

# ───────────────────────────────────────
# data/ 폴더 읽기
# ───────────────────────────────────────
def load_knowledge_base() -> str:
    data_dir = "data"
    if not os.path.exists(data_dir):
        return "세특 예시 데이터 없음"
    knowledge = ""
    for filepath in sorted(glob.glob(f"{data_dir}/*.txt")):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        knowledge += f"\n\n=== {filename} ===\n{content}"
    return knowledge.strip() or "세특 예시 데이터 없음"

KNOWLEDGE_BASE = load_knowledge_base()
print(f"✅ 세특 예시 데이터 로드 완료 ({len(KNOWLEDGE_BASE)} 글자)")

# ───────────────────────────────────────
# 세션
# ───────────────────────────────────────
@dataclass
class StudentSession:
    session_id:     str
    student_code:   Optional[str] = None
    student_name:   Optional[str] = None
    grade:          Optional[str] = None
    setech_type:    Optional[str] = None
    char_limit:     Optional[str] = None
    desired_career: Optional[str] = None
    desired_major:  Optional[str] = None
    mode:           Optional[str] = None
    result_setech:  Optional[str] = None
    result_comment: Optional[str] = None

_sessions: dict = {}

def get_or_create_session(session_id: str) -> StudentSession:
    if session_id not in _sessions:
        _sessions[session_id] = StudentSession(session_id=session_id)
    return _sessions[session_id]

# ───────────────────────────────────────
# Supabase 헬퍼
# ───────────────────────────────────────
def db_get_student(code: str) -> Optional[dict]:
    try:
        res = supabase.table("students").select("*").eq("code", code.upper()).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"학생 조회 오류: {e}")
        return None

def db_check_call_limit(code: str) -> tuple[bool, int, int]:
    try:
        student = db_get_student(code)
        if not student:
            return False, 0, 0
        limit = student.get("call_limit") or 0
        count = student.get("call_count") or 0
        if limit == 0:
            return True, count, limit
        return count < limit, count, limit
    except Exception as e:
        print(f"호출 제한 확인 오류: {e}")
        return True, 0, 0

def db_increment_call_count(code: str):
    try:
        student = db_get_student(code)
        if not student:
            return
        count = (student.get("call_count") or 0) + 1
        supabase.table("students").update({"call_count": count}).eq("code", code.upper()).execute()
    except Exception as e:
        print(f"호출 횟수 업데이트 오류: {e}")

def db_save_result(session: StudentSession, material: str):
    try:
        data = {
            "student_code":   session.student_code,
            "student_name":   session.student_name,
            "grade":          session.grade or "",
            "setech_type":    session.setech_type or "",
            "char_limit":     session.char_limit or "",
            "career":         session.desired_career or "",
            "major":          session.desired_major or "",
            "mode":           session.mode or "",
            "material":       material[:2000] if material else "",
            "result_setech":  session.result_setech or "",
            "result_comment": session.result_comment or "",
            "updated_at":     datetime.utcnow().isoformat(),
        }
        existing = supabase.table("setech_results").select("id").eq("student_code", session.student_code.upper()).execute()
        if existing.data:
            supabase.table("setech_results").update(data).eq("student_code", session.student_code.upper()).execute()
        else:
            supabase.table("setech_results").insert(data).execute()
    except Exception as e:
        print(f"결과 저장 오류: {e}")

# ───────────────────────────────────────
# Gemini 호출
# ───────────────────────────────────────
def build_system() -> str:
    return f"""{CORE_PRINCIPLES}

당신은 대한민국 최고의 입시 컨설턴트이자 학교생활기록부 세부능력특기사항 전문 작가입니다.
아래 세특 예시 데이터를 참고해서 핵심 원칙을 철저히 지키며 세특을 작성하세요.

[세특 예시 데이터 - 참고용]
{KNOWLEDGE_BASE}

[출력 형식 - 반드시 이 형식 그대로 출력]
=== 세특 ===
(완성된 세특 본문만. 마크다운 기호 없이)

=== 작성 포인트 ===
1. (어떤 부분을 왜 이렇게 썼는지 구체적으로)
2. (두 번째 포인트)
3. (세 번째 포인트)
(필요한 만큼 계속)"""

def build_user_msg(mode: str, setech_type: str, char_limit: str,
                   career: str, major: str, grade: str, content: str) -> str:
    base = f"""학년: {grade or '미입력'}
세특 종류: {setech_type}
분량 제한: {char_limit} (반드시 이 분량에 맞게 작성)
희망 진로: {career}
희망 학과: {major}
"""
    if mode == "rewrite":
        return base + f"""
[기존 세특]
{content}

위 세특을 입시 컨설턴트 관점에서 핵심 원칙에 따라 개선해주세요.
진로·희망학과 연결을 강화하고, 구체적인 활동 서술과 사고 과정이 잘 드러나도록 수정하세요.
분량 제한({char_limit})을 반드시 맞춰주세요.
"""
    else:
        return base + f"""
[활동 내용/자료]
{content}

위 활동 내용을 바탕으로 핵심 원칙에 따라 세특을 처음부터 작성해주세요.
진로·희망학과 연결이 자연스럽게 드러나도록 작성하세요.
분량 제한({char_limit})을 반드시 맞춰주세요.
"""

def parse_result(text: str) -> tuple[str, str]:
    if "=== 세특 ===" in text and "=== 작성 포인트 ===" in text:
        parts   = text.split("=== 작성 포인트 ===")
        setech  = parts[0].replace("=== 세특 ===", "").strip()
        comment = parts[1].strip() if len(parts) > 1 else ""
    elif "=== 세특 ===" in text:
        setech  = text.replace("=== 세특 ===", "").strip()
        comment = ""
    else:
        setech  = text.strip()
        comment = ""
    return setech, comment

def call_text(user_msg: str, student_code: str = None) -> str:
    if student_code:
        allowed, count, limit = db_check_call_limit(student_code)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"이용 횟수를 모두 사용했어요. (사용: {count}/{limit}회)")
        db_increment_call_count(student_code)
    response = client_genai.models.generate_content(
        model=MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(system_instruction=build_system())
    )
    return response.text

def call_vision(image_bytes: bytes, mime_type: str, prompt: str, student_code: str = None) -> str:
    if student_code:
        allowed, count, limit = db_check_call_limit(student_code)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"이용 횟수를 모두 사용했어요. (사용: {count}/{limit}회)")
        db_increment_call_count(student_code)
    response = client_genai.models.generate_content(
        model=MODEL,
        contents=[
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
            types.Part(text=prompt)
        ],
        config=types.GenerateContentConfig(system_instruction=build_system())
    )
    return response.text

# ───────────────────────────────────────
# FastAPI
# ───────────────────────────────────────
app = FastAPI(title="세특 AI 코치")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "세특 AI 코치 정상 작동 중"}

# ── 로그인 ──
class LoginRequest(BaseModel):
    student_code: str

@app.post("/login")
def login(req: LoginRequest):
    student = db_get_student(req.student_code)
    if not student:
        raise HTTPException(status_code=404, detail="등록되지 않은 코드예요.")
    session_id = str(uuid.uuid4())
    session = get_or_create_session(session_id)
    session.student_code = student["code"]
    session.student_name = student.get("name", "")
    session.grade        = str(student.get("grade", ""))
    _sessions[session_id] = session
    return {
        "session_id": session_id,
        "name":       session.student_name,
        "grade":      session.grade,
        "call_limit": student.get("call_limit") or 0,
        "call_count": student.get("call_count") or 0,
    }

# ── 세특 생성 (텍스트) ──
class GenerateTextRequest(BaseModel):
    session_id:     str
    setech_type:    str
    char_limit:     str
    desired_career: str
    desired_major:  str
    mode:           str
    content:        str

@app.post("/generate-text")
def generate_text(req: GenerateTextRequest):
    session = get_or_create_session(req.session_id)
    session.setech_type    = req.setech_type
    session.char_limit     = req.char_limit
    session.desired_career = req.desired_career
    session.desired_major  = req.desired_major
    session.mode           = req.mode
    _sessions[req.session_id] = session

    user_msg = build_user_msg(
        req.mode, req.setech_type, req.char_limit,
        req.desired_career, req.desired_major,
        session.grade or "", req.content
    )
    raw = call_text(user_msg, student_code=session.student_code)
    setech, comment = parse_result(raw)

    session.result_setech  = setech
    session.result_comment = comment
    _sessions[req.session_id] = session
    db_save_result(session, req.content)

    student = db_get_student(session.student_code) if session.student_code else {}
    return {
        "status":     "success",
        "setech":     setech,
        "comment":    comment,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0,
    }

# ── 세특 생성 (이미지) ──
@app.post("/generate-image")
async def generate_image(
    session_id:     str = Form(...),
    setech_type:    str = Form(...),
    char_limit:     str = Form(...),
    desired_career: str = Form(...),
    desired_major:  str = Form(...),
    mode:           str = Form(...),
    image: UploadFile = File(...),
):
    session = get_or_create_session(session_id)
    session.setech_type    = setech_type
    session.char_limit     = char_limit
    session.desired_career = desired_career
    session.desired_major  = desired_major
    session.mode           = mode
    _sessions[session_id]  = session

    image_bytes = await image.read()
    ext       = image.filename.split(".")[-1].lower() if image.filename else "jpg"
    mime_map  = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    task_desc = (
        "이미지의 세특을 입시 컨설턴트 관점에서 핵심 원칙에 따라 개선해주세요. 진로·희망학과 연결을 강화하고, 구체적인 활동 서술과 사고 과정이 잘 드러나도록 수정하세요."
        if mode == "rewrite" else
        "이미지의 활동 내용을 바탕으로 핵심 원칙에 따라 세특을 처음부터 작성해주세요. 진로·희망학과 연결이 자연스럽게 드러나도록 작성하세요."
    )
    prompt = f"""학년: {session.grade or '미입력'}
세특 종류: {setech_type}
분량 제한: {char_limit} (반드시 이 분량에 맞게 작성)
희망 진로: {desired_career}
희망 학과: {desired_major}

{task_desc}
분량 제한({char_limit})을 반드시 맞춰주세요.
"""
    raw = call_vision(image_bytes, mime_type, prompt, student_code=session.student_code)
    setech, comment = parse_result(raw)

    session.result_setech  = setech
    session.result_comment = comment
    _sessions[session_id]  = session
    db_save_result(session, "(이미지 업로드)")

    student = db_get_student(session.student_code) if session.student_code else {}
    return {
        "status":     "success",
        "setech":     setech,
        "comment":    comment,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

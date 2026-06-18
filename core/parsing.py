"""
Parsing layer — owned by the parsing workstream.

raw resume/JD text (or PDF/DOCX file) -> ParsedResume / ParsedJD

Uses Groq (free API tier, OpenAI-compatible) for LLM-assisted structured
extraction. Set GROQ_API_KEY in the environment. Output MUST match the
schema models in core/schema.py exactly.
"""
import json
import os
import time
from typing import Type

import docx
import pypdf
from groq import Groq
from pydantic import BaseModel, ValidationError

from .schema import ParsedJD, ParsedResume

# Free, capable model on Groq's hosted tier. Override with GROQ_MODEL if needed.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily build the Groq client so importing this module never crashes."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        _client = Groq(api_key=api_key)
    return _client


# ---------- File text extraction ----------

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()


def extract_text(file_path: str) -> str:
    """Extracts text from a file based on its extension (.pdf/.docx/.txt)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"Unsupported file extension: {ext}")


# ---------- LLM-assisted structured extraction ----------

def _llm_parse(text: str, model_cls: Type[BaseModel], retries: int = 2) -> dict:
    """Call Groq in JSON mode and return parsed JSON matching model_cls's schema."""
    client = _get_client()

    # Strip raw_text from the schema we show the model: we always overwrite it
    # ourselves afterward, and asking the model to echo the entire résumé back
    # into raw_text blows past max_tokens and breaks JSON generation.
    schema = model_cls.model_json_schema()
    schema.get("properties", {}).pop("raw_text", None)
    if isinstance(schema.get("required"), list):
        schema["required"] = [f for f in schema["required"] if f != "raw_text"]
    schema_json = json.dumps(schema)

    prompt = (
        "Extract structured information from the text below into JSON that "
        "strictly matches this JSON schema:\n"
        f"{schema_json}\n\n"
        "Rules: only use information present in the text; use null / empty "
        "lists when something is absent; do not invent skills or requirements. "
        "Do NOT include a 'raw_text' field — omit it entirely. Keep lists "
        "concise (the most relevant items only).\n\n"
        f"Text:\n{text}"
    )

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1024,
            )
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, Exception) as err:  # noqa: BLE001
            last_err = err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM parse failed after {retries + 1} attempts: {last_err}")


def _slim_schema(model_cls: Type[BaseModel]) -> dict:
    """Schema with raw_text removed (we attach raw_text ourselves afterward)."""
    schema = model_cls.model_json_schema()
    schema.get("properties", {}).pop("raw_text", None)
    if isinstance(schema.get("required"), list):
        schema["required"] = [f for f in schema["required"] if f != "raw_text"]
    return schema


def _cap(text: str, n: int = 4000) -> str:
    """Trim very long text so the parse prompt stays bounded."""
    text = text or ""
    return text if len(text) <= n else text[:n] + " …[truncated]"


def parse_both(resume_text: str, jd_text: str, retries: int = 2) -> tuple[ParsedResume, ParsedJD]:
    """
    Parse résumé AND job description in a SINGLE Groq call.

    Halves the number of LLM round-trips per analysis (4 calls -> 2) so the whole
    pipeline fits inside the free tier's per-minute token budget. The full,
    uncapped source text is still preserved as each model's raw_text.
    """
    client = _get_client()
    resume_schema = json.dumps(_slim_schema(ParsedResume))
    jd_schema = json.dumps(_slim_schema(ParsedJD))

    prompt = (
        "Extract structured data from BOTH the résumé and the job description "
        "below. Return ONE JSON object with exactly two keys: 'resume' and 'job'.\n\n"
        f"'resume' must match this JSON schema:\n{resume_schema}\n\n"
        f"'job' must match this JSON schema:\n{jd_schema}\n\n"
        "Rules: only use information present in each text; use null / empty lists "
        "when something is absent; do not invent skills or requirements. Do NOT "
        "include any 'raw_text' field — omit it entirely. Keep lists concise.\n\n"
        f"=== RÉSUMÉ ===\n{_cap(resume_text)}\n\n"
        f"=== JOB DESCRIPTION ===\n{_cap(jd_text)}"
    )

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1536,
            )
            data = json.loads(response.choices[0].message.content)
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"LLM parse failed after {retries + 1} attempts: {last_err}")

    rdata = data.get("resume") or {}
    jdata = data.get("job") or {}
    rdata["raw_text"] = resume_text  # preserve the full, uncapped source text
    jdata["raw_text"] = jd_text
    if not str(jdata.get("title") or "").strip():
        jdata["title"] = "Unknown role"  # ParsedJD.title is required

    try:
        return ParsedResume(**rdata), ParsedJD(**jdata)
    except ValidationError as err:
        raise ValueError(f"Parsed data did not match the schema: {err}") from err


def parse_resume(text: str) -> ParsedResume:
    """Parses resume text into a ParsedResume model."""
    data = _llm_parse(text, ParsedResume)
    data["raw_text"] = text  # always preserve the source text
    try:
        return ParsedResume(**data)
    except ValidationError as err:
        raise ValueError(f"Resume did not match ParsedResume schema: {err}") from err


def parse_jd(text: str) -> ParsedJD:
    """Parses JD text into a ParsedJD model."""
    data = _llm_parse(text, ParsedJD)
    data["raw_text"] = text  # always preserve the source text
    try:
        return ParsedJD(**data)
    except ValidationError as err:
        raise ValueError(f"JD did not match ParsedJD schema: {err}") from err

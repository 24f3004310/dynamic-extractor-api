import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI()

# --- Rule 4: Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The request body accepts text and an unpredictable, dynamic schema dictionary
class DynamicExtractRequest(BaseModel):
    text: str
    schema_def: Dict[str, str] = Field(..., alias="schema") 
    # 'alias' maps the incoming JSON key "schema" to "schema_def" safely in Python

client = Groq()

def clean_and_cast_types(extracted_data: Dict[str, Any], schema_def: Dict[str, str]) -> Dict[str, Any]:
    """
    Enforces Rule 1, 2, and 3: Sanitizes LLM output to match requested types perfectly.
    """
    final_output = {}

    for key, expected_type in schema_def.items():
        value = extracted_data.get(key, None)

        # Rule 2: If field is missing or looks like null, set to None
        if value is None or str(value).lower() in ["null", "none", "na", "n/a", ""]:
            final_output[key] = None
            continue

        try:
            if expected_type == "integer":
                final_output[key] = int(float(str(value)))
            elif expected_type == "float":
                final_output[key] = float(str(value))
            elif expected_type == "boolean":
                val_str = str(value).lower()
                final_output[key] = val_str in ["true", "1", "yes"]
            elif expected_type == "date":
                # Ensure date follows ISO format YYYY-MM-DD
                # If LLM didn't format it right, we fallback to a safe string or null
                date_str = str(value).split("T")[0].strip() # Handles potential ISO timestamps
                final_output[key] = date_str
            else:
                # Default to string
                final_output[key] = str(value)
        except Exception:
            # Fallback gracefully to null if conversion fails completely
            final_output[key] = None

    return final_output

@app.post("/dynamic-extract")
async def dynamic_extract(payload: DynamicExtractRequest):
    try:
        if not os.environ.get("GROQ_API_KEY"):
            return {"error": "GROQ_API_KEY is missing from environment variables"}

        # Tell the LLM exactly how to format the dynamic schema
        system_prompt = (
            "You are an adaptive data extraction engine.\n"
            "Analyze the provided text and extract ONLY the fields requested in the user-provided schema.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Your response must be a single, valid JSON object containing exactly these keys: {list(payload.schema_def.keys())}.\n"
            f"2. Match the requested data types specified in the schema: {json.dumps(payload.schema_def)}.\n"
            f"3. For fields of type 'date', return them strictly as 'YYYY-MM-DD'.\n"
            f"4. If a field cannot be found or inferred from the text, return null for that key.\n"
            f"5. Do NOT include any explanations, markdown code blocks, or extra text. Output raw JSON only."
        )

        # Execute high-speed inference with Groq's smart 70B model
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to extract from: {payload.text}"}
            ],
            response_format={"type": "json_object"}, # Strictly forces valid JSON structure
            temperature=0.0
        )

        # Parse the raw LLM response
        raw_response = chat_completion.choices[0].message.content.strip()
        parsed_llm_json = json.loads(raw_response)

        # Run our post-processing cleaner to guarantee typing rules & remove extra keys
        validated_output = clean_and_cast_types(parsed_llm_json, payload.schema_def)

        return validated_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Dynamic Extractor Engine is active"}
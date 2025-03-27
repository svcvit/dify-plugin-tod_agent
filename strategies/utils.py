import json
import re
from typing import Optional
from dify_plugin.entities.model.llm import LLMUsage

def try_parse_json(content: str) -> dict:
    try:
        extracted_data = json.loads(content)
        return extracted_data
    except json.JSONDecodeError:
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group(0))
                return extracted_data
        except Exception:
            pass
        return {}

def increase_usage(
    llm_usage: dict[str, Optional[LLMUsage]], 
    usage: Optional[LLMUsage]
) -> None:
    if usage is None:
        return
    if llm_usage["usage"] is None:
        llm_usage["usage"] = usage
    else:
        llm_usage["usage"].prompt_tokens += usage.prompt_tokens
        llm_usage["usage"].completion_tokens += usage.completion_tokens
        llm_usage["usage"].total_tokens += usage.total_tokens
        llm_usage["usage"].total_price += usage.total_price
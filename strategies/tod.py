import json
import time
from typing import Any, Generator, Optional

from pydantic import BaseModel, field_validator

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import (
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.interfaces.agent import AgentStrategy, AgentModelConfig
from dify_plugin.entities.tool import ToolInvokeMessage, LogMetadata


class DialogueField(BaseModel):
    name: str
    question: str
    required: bool = True
    value: Optional[str] = None


class DialogueState(BaseModel):
    current_field_index: int = 0
    fields: list[DialogueField]
    completed: bool = False


class TODParams(BaseModel):
    information_schema: str
    query: str
    model: AgentModelConfig
    storage_key: str

    @field_validator("information_schema")
    def validate_information_schema(cls, v):
        try:
            schema = json.loads(v)
            if not isinstance(schema, dict) or "fields" not in schema:
                raise ValueError("Invalid schema format")
            return v
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")


class TaskOrientedDialogueStrategy(AgentStrategy):
    def __init__(self, session):
        super().__init__(session)
        self.dialogue_state = None
        self.collected_data = {}
        self.current_model_config = None
        self.llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}

    def _get_conversation_history(self) -> str:
        history = []
        for field in self.dialogue_state.fields[
            : self.dialogue_state.current_field_index
        ]:
            if field.value:
                history.append(f"Q: {field.question}\nA: {field.value}")
        return "\n".join(history)

    def _init_dialogue_state(self, information_schema: str) -> DialogueState:
        try:
            schema = json.loads(information_schema)
            fields = []
            for field in schema.get("fields", []):
                fields.append(
                    DialogueField(
                        name=field["name"],
                        question=field["question"],
                        required=field.get("required", True),
                    )
                )
            return DialogueState(fields=fields)
        except Exception:
            raise ValueError("Failed to initialize dialogue state")

    def _load_dialogue_state(self, storage_key: str) -> Optional[DialogueState]:
        try:
            if stored_data := self.session.storage.get(storage_key):
                state_dict = json.loads(stored_data.decode())
                dialogue_state = DialogueState(**state_dict)
                self.collected_data = {
                    field.name: field.value
                    for field in dialogue_state.fields
                    if field.value is not None
                }
                return dialogue_state
        except Exception:
            return None

    def _save_dialogue_state(self, storage_key: str):
        try:
            if self.dialogue_state:
                state_dict = self.dialogue_state.model_dump()
                self.session.storage.set(storage_key, json.dumps(state_dict).encode())
        except Exception:
            pass

    def _get_system_prompt(self) -> str:
        return """You are a professional conversation assistant. Your task is to evaluate whether the user's response meets the requirements of the current question.

Please evaluate according to the following rules:
1. If the user's answer fully meets the question requirements, please directly reply with the user's answer
2. If the user's answer contains both the answer to the current question and answers to other questions, it is also considered valid, please directly reply with the user's answer
3. If the user is greeting or making small talk, please reply with "INVALID: User is greeting or making small talk"
4. If the user's answer is incomplete or unclear, please reply with "INVALID: " followed by the specific reason
5. If the user's answer is irrelevant to the question, please reply with "INVALID: Answer is irrelevant to the question"

Examples:
Q: What is your contact number?
A: Hello
Response: INVALID: User is greeting or making small talk

Q: What is your contact number?
A: My phone
Response: INVALID: Answer is incomplete, need specific number

Q: What is your contact number?
A: 13812345678
Response: 13812345678

Q: Where do you want to travel?
A: I want to spend 30,000 to travel to Japan for 5 days
Response: Japan"""

    def _extract_answers(self, user_input: str, parent_log=None) -> Generator[AgentInvokeMessage, None, None]:
        extract_prompt = f"""Please extract all relevant information from the user's response and return it in JSON format.  Return the extracted information in the same language as the user's input.

    Rules:
    1. Carefully analyze user input to extract information related to all questions
    2. Even if the user only answered one question, try to infer answers to other questions from the context
    3. If certain information cannot be extracted or inferred, do not include that field
    4. The return format must be valid JSON

    Current information to collect:
    {[{"name": field.name, "question": field.question} for field in self.dialogue_state.fields]}

    User input: {user_input}

    Please return in JSON format, for example:
    {{"destination": "Japan", "duration": "5 days", "budget": "30,000"}}

    Note: Even if the user only answered some questions, please try to extract all relevant information.  The output language MUST match the user's input language."""

        extract_started_at = time.perf_counter()
        extract_log = self.create_log_message(
            label=f"Answer Extraction",
            data={
                "user_input": user_input,
                "extract_prompt": extract_prompt,
                "type": "extraction"
            },
            metadata={
                LogMetadata.STARTED_AT: extract_started_at,
            },
            parent=parent_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        yield extract_log

        model_config = LLMModelConfig(**self.current_model_config.model_dump(mode="json"))
        response = self.session.model.llm.invoke(
            model_config=model_config,
            prompt_messages=[
                SystemPromptMessage(content=extract_prompt)
            ],
            stream=False
        )
        extracted_data = self._try_parse_json(response.message.content)

        # Move increase_usage here, after the LLM call in extraction
        self.increase_usage(self.llm_usage, response.usage)

        finish_log = self.finish_log_message(
            log=extract_log,
            data={
                "response": response.message.content,
                "extracted_data": extracted_data
            },
            metadata={
                LogMetadata.STARTED_AT: extract_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - extract_started_at,
                LogMetadata.TOTAL_TOKENS: response.usage.total_tokens if response.usage else 0,
            }
        )
        yield finish_log
        yield extracted_data

    def _try_parse_json(self, content: str) -> dict:
        try:
            extracted_data = json.loads(content)
            return extracted_data
        except json.JSONDecodeError:
            try:
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group(0))
                    return extracted_data
            except Exception:
                pass
            return {}

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage, None, None]:
        params = TODParams(**parameters)
        self.current_model_config = params.model
        self.llm_usage = {"usage": None}

        round_started_at = time.perf_counter()
        round_log = self.create_log_message(
            label=f"Dialogue Round",
            data={
                "query": params.query,
                "information_schema": params.information_schema,
                "conversation_history": self._get_conversation_history() if self.dialogue_state else "",
                "dialogue_state": {
                    "current_field_index": self.dialogue_state.current_field_index if self.dialogue_state else 0,
                    "total_fields": len(self.dialogue_state.fields) if self.dialogue_state else 0,
                    "completed_fields": [
                        {"name": f.name, "question": f.question, "value": f.value}
                        for f in self.dialogue_state.fields
                        if self.dialogue_state and f.value is not None
                    ]
                } if self.dialogue_state else None
            },
            metadata={
                LogMetadata.STARTED_AT: round_started_at,
            },
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        yield round_log

        if not self.dialogue_state:
            try:
                self.dialogue_state = self._load_dialogue_state(params.storage_key)
            except Exception:
                self.dialogue_state = None

            if not self.dialogue_state:
                self.dialogue_state = self._init_dialogue_state(params.information_schema)

        current_field = self.dialogue_state.fields[self.dialogue_state.current_field_index]

        if params.query.strip() == "":
            message = current_field.question
            yield self.create_text_message(message)

            finish_log_data = {
                "output": message,
                "dialogue_state": {
                    "current_field": current_field.name,
                    "current_field_index": self.dialogue_state.current_field_index,
                    "total_fields": len(self.dialogue_state.fields),
                    "completed_fields": [
                        {"name": f.name, "value": f.value}
                        for f in self.dialogue_state.fields
                        if f.value is not None
                    ]
                }
            }
            finish_log_metadata = {
                LogMetadata.STARTED_AT: round_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                LogMetadata.TOTAL_TOKENS: 0,
            }

            yield self.finish_log_message(
                log=round_log,
                data=finish_log_data,
                metadata=finish_log_metadata
            )
            return

        conversation_history = self._get_conversation_history()
        context_prompt = f"""Collected information:
{conversation_history}

Current question: {current_field.question}
User answer: {params.query}"""

        system_message = SystemPromptMessage(content=self._get_system_prompt())

        validation_started_at = time.perf_counter()
        validation_log = self.create_log_message(
            label=f"Answer Validation",
            data={
                "context": context_prompt,
                "system_prompt": system_message.content,
                "type": "validation"
            },
            metadata={
                LogMetadata.STARTED_AT: validation_started_at,
                LogMetadata.PROVIDER: params.model.provider,
            },
            parent=round_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        yield validation_log

        model_config = LLMModelConfig(**params.model.model_dump(mode="json"))
        response = self.session.model.llm.invoke(
            model_config=model_config,
            prompt_messages=[
                system_message,
                UserPromptMessage(content=context_prompt),
            ],
            stream=False
        )

        is_valid = self._is_valid_answer(response.message.content)
        # Move increase_usage here, after the LLM call in validation
        self.increase_usage(self.llm_usage, response.usage)

        yield self.finish_log_message(
            log=validation_log,
            data={
                "response": response.message.content,
                "is_valid": is_valid
            },
            metadata={
                LogMetadata.STARTED_AT: validation_started_at,
                LogMetadata.FINISHED_AT: time.perf_counter(),
                LogMetadata.ELAPSED_TIME: time.perf_counter() - validation_started_at,
                LogMetadata.PROVIDER: params.model.provider,
                LogMetadata.TOTAL_TOKENS: response.usage.total_tokens if response.usage else 0,
            }
        )

        extracted_answers = {}
        if is_valid or params.query.strip() != "":
            extraction_generator = self._extract_answers(params.query, round_log)
            last_item = None
            for item in extraction_generator:
                if isinstance(item, AgentInvokeMessage):
                    yield item
                else:
                    last_item = item

            if isinstance(last_item, dict):
                extracted_answers = last_item

            answers_saved = False
            fields_to_update = {}

            for field in self.dialogue_state.fields:
                if field.name in extracted_answers and extracted_answers[field.name]:
                    fields_to_update[field.name] = extracted_answers[field.name]
                    answers_saved = True

            if answers_saved:
                for field in self.dialogue_state.fields:
                    if field.name in fields_to_update:
                        field.value = fields_to_update[field.name]
                        self.collected_data[field.name] = field.value

                next_field_index = len(self.dialogue_state.fields)
                for i, field in enumerate(self.dialogue_state.fields):
                    if field.value is None or field.value == "":
                        next_field_index = i
                        break

                self.dialogue_state.current_field_index = next_field_index
                self._save_dialogue_state(params.storage_key)

            if self.dialogue_state.current_field_index >= len(self.dialogue_state.fields):
                self.dialogue_state.completed = True
                summary = self._generate_summary()
                yield self.create_text_message("InformationCollectionCompleted:\n"+summary)
                yield self.create_json_message(self.collected_data)

                finish_log_data = {
                    "output": summary,
                    "collected_data": self.collected_data,
                    "dialogue_state": {
                        "completed": True,
                        "current_field_index": self.dialogue_state.current_field_index,
                        "total_fields": len(self.dialogue_state.fields),
                        "completed_fields": [
                            {"name": f.name, "value": f.value}
                            for f in self.dialogue_state.fields
                        ]
                    }
                }
                finish_log_metadata = {
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                }

                # 在结束前也添加执行元数据的输出
                yield self.create_json_message(
                    {
                        "execution_metadata": {
                            LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price
                            if self.llm_usage["usage"] is not None
                            else 0,
                            LogMetadata.CURRENCY: self.llm_usage["usage"].currency
                            if self.llm_usage["usage"] is not None
                            else "",
                            LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens
                            if self.llm_usage["usage"] is not None
                            else 0,
                        }
                    }
                )

                yield self.finish_log_message(
                    log=round_log,
                    data=finish_log_data,
                    metadata=finish_log_metadata
                )
                self.session.storage.delete(params.storage_key)
                return
            else:
                next_field = self.dialogue_state.fields[self.dialogue_state.current_field_index]
                message = next_field.question
                yield self.create_text_message(message)
        else:
            reason = response.message.content.split("INVALID:", 1)[1].strip() if response.message.content.startswith("INVALID:") else "回答不够明确"

            if "greeting" in reason or "small talk" in reason.lower():
                message = f"Hello! {current_field.question}"
            elif "incomplete" in reason or "unclear" in reason.lower():
                message = f"{reason}, please provide {current_field.question}"
            else:
                message = f"{reason}, {current_field.question}"
            yield self.create_text_message(message)

        yield self.create_json_message(
            {
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price
                    if self.llm_usage["usage"] is not None
                    else 0,
                    LogMetadata.CURRENCY: self.llm_usage["usage"].currency
                    if self.llm_usage["usage"] is not None
                    else "",
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens
                    if self.llm_usage["usage"] is not None
                    else 0,
                }
            }
        )

        finish_log_data = {
            "output": message,
            "dialogue_state": {
                "completed": False,
                "current_field_index": self.dialogue_state.current_field_index,
                "total_fields": len(self.dialogue_state.fields),
                "completed_fields": [
                    {"name": f.name, "value": f.value}
                    for f in self.dialogue_state.fields
                    if f.value is not None
                ],
                "current_field": {
                    "name": current_field.name,
                    "question": current_field.question
                }
            },
            "model_interactions_summary": {
                "validation_result": is_valid,
                "extracted_answers": extracted_answers,
                "total_interactions": 2
            }
        }
        finish_log_metadata = {
            LogMetadata.STARTED_AT: round_started_at,
            LogMetadata.FINISHED_AT: time.perf_counter(),
            LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
            LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
        }

        yield self.finish_log_message(
            log=round_log,
            data=finish_log_data,
            metadata=finish_log_metadata
        )

    def _is_valid_answer(self, model_response: str) -> bool:
        try:
            if model_response.startswith("INVALID:"):
                return False

            return True

        except Exception:
            return False

    def _generate_summary(self) -> str:
        summary = ""
        for field in self.dialogue_state.fields:
            summary += f"{field.question} {field.value}\n"
        return summary.strip()

    def increase_usage(
        self, llm_usage: dict[str, Optional[LLMUsage]], usage: Optional[LLMUsage]
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
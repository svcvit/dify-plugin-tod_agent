import json
import time
from typing import Any, Generator, Optional

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from dify_plugin.interfaces.agent import AgentStrategy
from dify_plugin.entities.tool import ToolInvokeMessage, LogMetadata

from .models import DialogueState, DialogueField, TODParams
from .utils import try_parse_json, increase_usage
from .prompts import tod_system_prompt, tod_extract_prompt

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

    def _extract_answers(self, user_input: str, parent_log=None) -> Generator[AgentInvokeMessage, None, None]:
        extract_prompt = tod_extract_prompt(self.dialogue_state.fields, user_input)

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
        extracted_data = try_parse_json(response.message.content)

        increase_usage(self.llm_usage, response.usage)

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

        system_message = SystemPromptMessage(content=tod_system_prompt())

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
        increase_usage(self.llm_usage, response.usage)

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

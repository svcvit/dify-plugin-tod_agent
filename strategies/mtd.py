"""
Implements the Multi-Turn Dialogue (MTD) strategy for agents.

This strategy guides a conversation based on predefined instructions,
handling multiple turns, state management, and determining completion.
"""

import json
import re # <<< ADDED IMPORT
import time
from typing import Any, Generator, Optional

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage, LLMResult
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from dify_plugin.interfaces.agent import AgentStrategy, AgentModelConfig
from dify_plugin.entities.tool import ToolInvokeMessage, LogMetadata

from .models import MTDParams, MTDContext, DialogueHistory
from .utils import increase_usage
from .prompts import mtd_system_prompt

class MultiTurnDialogueStrategy(AgentStrategy):
    """
    Manages a multi-turn dialogue based on provided instructions.

    It handles loading/saving dialogue context, generating the first question,
    processing user responses using an LLM, determining the next question or
    dialogue completion according to instructions, and logging the process.
    """
    def __init__(self, session):
        super().__init__(session)
        self.context: Optional[MTDContext] = None
        self.current_model_config: Optional[AgentModelConfig] = None
        self.llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}


    def _load_context(self, storage_key: str, instruction: str) -> MTDContext:
        """Loads dialogue context from storage or initializes a new one."""
        try:
            if stored_data := self.session.storage.get(storage_key):
                # Ensure loaded context has the potentially updated instruction
                loaded_dict = json.loads(stored_data.decode())
                loaded_dict['instruction'] = instruction # Overwrite with current instruction
                return MTDContext(**loaded_dict)
        except Exception as e:
            # TODO: Replace print with proper logging (e.g., self.session.logger or standard logging)
            print(f"Failed to load MTD context for key {storage_key}: {e}") # Simple print for debugging
            pass
        # If loading fails or no data, create a new context
        return MTDContext(instruction=instruction)

    def _save_context(self, storage_key: str):
        """Saves the current dialogue context to storage."""
        try:
            if self.context:
                self.session.storage.set(
                    storage_key,
                    json.dumps(self.context.model_dump()).encode()
                )
        except Exception as e:
             # TODO: Replace print with proper logging
             print(f"Failed to save MTD context for key {storage_key}: {e}") # Simple print for debugging
             pass

    def _get_first_question(self, model_config: LLMModelConfig, parent_log: ToolInvokeMessage.LogMessage) -> Generator[AgentInvokeMessage, None, str]:
        """
        Generates the first question using an LLM call based on instructions
        and yields log messages during the process. Returns the question string.
        """
        system_prompt_content = """You are a dialogue assistant. Based on the following multi-turn dialogue instructions, generate the first question:

Instructions:
{instruction}

Please return only the first question directly, without any explanation, greeting, or other content."""

        formatted_prompt = system_prompt_content.format(instruction=self.context.instruction)

        log_start_time = time.perf_counter()
        # Create the log message to be yielded (this IS the AgentInvokeMessage)
        first_q_log_msg = self.create_log_message(
            label="Get First Question",
            data={
                "instruction": self.context.instruction,
                "prompt": formatted_prompt,
                "type": "llm_call"
            },
            metadata={LogMetadata.STARTED_AT: log_start_time},
            parent=parent_log,
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        # Yield the START AgentInvokeMessage
        yield first_q_log_msg

        first_question = ""
        try:
            response: LLMResult = self.session.model.llm.invoke(
                model_config=model_config,
                prompt_messages=[SystemPromptMessage(content=formatted_prompt)],
                stream=False
            )
            increase_usage(self.llm_usage, response.usage)
            first_question = response.message.content.strip()

            log_finish_time = time.perf_counter()
            # Yield the FINISH AgentInvokeMessage
            yield self.finish_log_message(
                log=first_q_log_msg, # Pass the START message to correlate
                data={
                    "response": first_question,
                },
                metadata={
                    LogMetadata.STARTED_AT: log_start_time,
                    LogMetadata.FINISHED_AT: log_finish_time,
                    LogMetadata.ELAPSED_TIME: log_finish_time - log_start_time,
                    LogMetadata.TOTAL_TOKENS: response.usage.total_tokens if response.usage else 0,
                }
            )

        except Exception as e:
            log_finish_time = time.perf_counter()
            # Yield the ERROR FINISH AgentInvokeMessage
            yield self.finish_log_message(
                log=first_q_log_msg, # Pass the START message to correlate
                data={"error": str(e)},
                metadata={
                    LogMetadata.STARTED_AT: log_start_time,
                    LogMetadata.FINISHED_AT: log_finish_time,
                    LogMetadata.ELAPSED_TIME: log_finish_time - log_start_time,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.ERROR
            )
            # Error occurred, first_question remains ""
        finally:
            # Yield the result (question string or "" on error) after logging
            yield first_question


    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage, None, None]:
        """
        Executes one round of the multi-turn dialogue strategy.

        Loads context, handles the first turn or subsequent turns,
        interacts with the LLM to process user input, updates context,
        yields messages (questions, completion summary, errors), logs actions,
        and manages dialogue state.
        """
        params = MTDParams(**parameters)
        self.current_model_config = params.model
        self.llm_usage = {"usage": None} # Reset usage for each invocation

        round_started_at = time.perf_counter()

        # Load context *before* starting the round log to include initial state
        self.context = self._load_context(params.storage_key, params.instruction)

        # Start Round Log
        round_log_data = {
            "query": params.query,
            "instruction": params.instruction,
            "initial_context": self.context.model_dump() if self.context else None,
            "storage_key": params.storage_key,
        }
        # Create the round log message to be yielded
        round_log_msg = self.create_log_message(
            label="Multi-Turn Dialogue Round",
            data=round_log_data,
            metadata={LogMetadata.STARTED_AT: round_started_at},
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        # Yield the START AgentInvokeMessage for the round
        yield round_log_msg

        # --- Handle First Turn ---
        if not self.context.current_question:
            first_q_gen = self._get_first_question(
                LLMModelConfig(**self.current_model_config.model_dump(mode="json")),
                round_log_msg # Pass the START message as parent context
            )
            first_question = ""
            for item in first_q_gen:
                if isinstance(item, AgentInvokeMessage):
                    yield item # Yield log messages from _get_first_question
                else:
                    # This should be the final yield (string) from _get_first_question
                    first_question = item if isinstance(item, str) else ""

            if first_question:
                self.context.current_question = first_question
                self._save_context(params.storage_key)
                yield self.create_text_message(self.context.current_question)

                # Yield metadata for the first question generation
                yield self.create_json_message({
                    "execution_metadata": {
                        LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price if self.llm_usage["usage"] else 0,
                        LogMetadata.CURRENCY: self.llm_usage["usage"].currency if self.llm_usage["usage"] else "",
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                    }
                })

                # Finish round log for the first turn
                yield self.finish_log_message(
                    log=round_log_msg, # Pass the START message to correlate
                    data={
                        "output": self.context.current_question,
                        "final_context": self.context.model_dump(),
                        "status": "First question generated"
                    },
                    metadata={
                        LogMetadata.STARTED_AT: round_started_at,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                        LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                    }
                )
            else:
                # Handle error during first question generation
                error_message = "Sorry, there was a problem initializing the conversation. Please try again later."
                yield self.create_text_message(error_message)
                yield self.finish_log_message(
                    log=round_log_msg, # Pass the START message to correlate
                    data={"output": error_message, "status": "Error during initialization"},
                    metadata={
                        LogMetadata.STARTED_AT: round_started_at,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                         LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                    },
                    status=ToolInvokeMessage.LogMessage.LogStatus.ERROR
                )
            return # End execution for the first turn

        # --- Handle Subsequent Turns ---

        # If user sent empty query, just repeat the current question
        if not params.query.strip():
            yield self.create_text_message(self.context.current_question)
            # Yield metadata (usage is 0 for this path)
            yield self.create_json_message({
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: 0,
                    LogMetadata.CURRENCY: "",
                    LogMetadata.TOTAL_TOKENS: 0,
                }
            })
            # Finish round log
            yield self.finish_log_message(
                log=round_log_msg, # Pass the START message to correlate
                data={
                    "output": self.context.current_question,
                    "final_context": self.context.model_dump(),
                    "status": "Repeated question due to empty input"
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: 0, # No LLM call here
                }
            )
            return

        # --- Main LLM Call for Dialogue Logic ---
        history_text = "\n".join([f"Q: {h.question}\nA: {h.answer}" for h in self.context.history])
        system_prompt = mtd_system_prompt().format(
            instruction=self.context.instruction,
            history=history_text,
            current_question=self.context.current_question,
            user_input=params.query # Renamed for clarity in prompt
        )

        llm_start_time = time.perf_counter()
        # Create the dialogue log message to be yielded
        dialogue_log_msg = self.create_log_message(
            label="Process User Response",
            data={ # Initial data for START message
                "history": history_text,
                "current_question": self.context.current_question,
                "user_input": params.query,
                "prompt": system_prompt,
                "type": "llm_call"
            },
            metadata={LogMetadata.STARTED_AT: llm_start_time},
            parent=round_log_msg, # Use START round message as parent context
            status=ToolInvokeMessage.LogMessage.LogStatus.START
        )
        # Yield the START AgentInvokeMessage for dialogue processing
        yield dialogue_log_msg

        result = "" # Initialize result to handle potential errors early
        dialogue_log_status = ToolInvokeMessage.LogMessage.LogStatus.SUCCESS
        # Prepare data payload for the dialogue log's FINISH message
        dialogue_finish_data = {}
        response = None # Initialize response variable
        llm_finish_time = time.perf_counter() # Initialize finish time

        try:
            # --- LLM Interaction ---
            model_config_llm = LLMModelConfig(**self.current_model_config.model_dump(mode="json"))
            response = self.session.model.llm.invoke( # Assign to response variable
                model_config=model_config_llm,
                prompt_messages=[SystemPromptMessage(content=system_prompt)],
                stream=False
            )
            increase_usage(self.llm_usage, response.usage)
            result = response.message.content.strip()
            llm_finish_time = time.perf_counter() # Capture actual finish time

            # Base data for the dialogue log FINISH message
            dialogue_finish_data = {"response": result}

            # We don't yield the dialogue finish message immediately,
            # process the result first to determine final status and add warnings/errors.

            final_output_message = ""
            final_status = "Processing complete"
            final_context_state = self.context.model_dump()
            dialogue_should_complete = False

            # --- Process LLM Result ---
            if result.startswith("INVALID:"):
                reason = result.split("INVALID:", 1)[1].strip()
                final_output_message = f"{reason}\n{self.context.current_question}"
                yield self.create_text_message(final_output_message)
                final_status = f"Invalid user input: {reason}"
                dialogue_log_status = ToolInvokeMessage.LogMessage.LogStatus.ERROR # Treat invalid input as an error for logging status
                # Add error detail to the finish data
                dialogue_finish_data["error"] = f"Invalid user input: {reason}"


            elif result == "MTD_COMPLETED":
                dialogue_should_complete = True
                final_status = "Dialogue completed (explicit MTD_COMPLETED)"
                self.context.history.append(DialogueHistory(
                    question=self.context.current_question,
                    answer=params.query
                ))
                # dialogue_log_status remains SUCCESS

            else:
                # --- Attempt to Parse JSON (with Markdown fence extraction) ---
                potential_json_str = result
                markdown_extracted = False
                try:
                    # <<< MODIFICATION START: Extract from Markdown fences >>>
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result, re.DOTALL | re.IGNORECASE)
                    if match:
                        potential_json_str = match.group(1).strip()
                        markdown_extracted = True
                    # <<< MODIFICATION END >>>

                    # Try parsing the potential JSON string
                    cleaned_result = potential_json_str.strip('"') # Clean quotes just in case
                    # Check if it looks like a JSON object before parsing
                    if cleaned_result.startswith('{') and cleaned_result.endswith('}'):
                        result_json = json.loads(cleaned_result)

                        # Add warning if extracted from markdown
                        if markdown_extracted:
                            warning_msg = f"LLM returned JSON wrapped in Markdown fences. Extracted successfully."
                            if "warning" not in dialogue_finish_data: dialogue_finish_data["warning"] = []
                            dialogue_finish_data["warning"].append(warning_msg)
                            # Keep dialogue_log_status SUCCESS despite warning

                        # Process the valid JSON
                        next_question = result_json.get("next_question")
                        user_answer = result_json.get("answer", params.query)

                        if next_question:
                            # Advance to next question
                            self.context.history.append(DialogueHistory(question=self.context.current_question, answer=user_answer))
                            self.context.current_question = next_question
                            self._save_context(params.storage_key)
                            final_output_message = self.context.current_question
                            yield self.create_text_message(final_output_message)
                            final_status = "Advanced to next question"
                            final_context_state = self.context.model_dump()
                        else:
                            # Implicit Completion (JSON without next_question)
                            dialogue_should_complete = True
                            final_status = "Dialogue completed (JSON without next_question)"
                            self.context.history.append(DialogueHistory(question=self.context.current_question, answer=user_answer))
                    else:
                        # --- Format Error: Not JSON even after potential extraction ---
                        # <<< MODIFICATION START: Handle non-completion >>>
                        final_status = f"LLM response format error: Invalid structure '{result}'"
                        dialogue_log_status = ToolInvokeMessage.LogMessage.LogStatus.ERROR
                        error_msg = f"Received an unexpected response format from the assistant."
                        dialogue_finish_data["error"] = [error_msg + f" Raw: {result}"]
                        # Prepare message to user to re-ask
                        final_output_message = f"{error_msg} Could you please answer again: {self.context.current_question}"
                        yield self.create_text_message(final_output_message)
                        # Keep current context, do not add history for this failed step
                        # dialogue_should_complete remains False
                        # <<< MODIFICATION END >>>

                except json.JSONDecodeError as e:
                    # --- JSON Parsing Error ---
                    # <<< MODIFICATION START: Handle non-completion >>>
                    error_detail = str(e)
                    final_status = f"LLM response JSON parse error: {error_detail} on content '{potential_json_str}' (extracted from '{result}')"
                    dialogue_log_status = ToolInvokeMessage.LogMessage.LogStatus.ERROR
                    error_msg = f"Failed to process the assistant's response (JSON parse error)."
                    dialogue_finish_data["error"] = [error_msg + f" Detail: {error_detail}"]
                    # Prepare message to user to re-ask
                    final_output_message = f"{error_msg} Could you please answer again: {self.context.current_question}"
                    yield self.create_text_message(final_output_message)
                    # Keep current context, do not add history for this failed step
                    # dialogue_should_complete remains False
                    # <<< MODIFICATION END >>>


            # --- Now, yield the FINISH message for the dialogue processing step ---
            yield self.finish_log_message(
                log=dialogue_log_msg, # Correlate with the START message
                data=dialogue_finish_data, # Pass the constructed payload
                metadata={
                    LogMetadata.STARTED_AT: llm_start_time,
                    LogMetadata.FINISHED_AT: llm_finish_time,
                    LogMetadata.ELAPSED_TIME: llm_finish_time - llm_start_time,
                    LogMetadata.TOTAL_TOKENS: response.usage.total_tokens if response and response.usage else 0,
                },
                status=dialogue_log_status # Use the determined status
            )

            # --- Handle Dialogue Completion (if flagged in this round) ---
            if dialogue_should_complete:
                self.context.completed = True
                self.context.current_question = "" # Clear current question

                summary = "Multi-turn dialogue completed. Summary:\n" + "\n".join([
                    f"Question: {h.question}\nAnswer: {h.answer}" for h in self.context.history
                ])
                final_output_message = "MTD_COMPLETED:\n" + summary
                yield self.create_text_message(final_output_message)
                # Update final context state for logging
                final_context_state = self.context.model_dump()
                # Clean up storage ONLY on successful completion path
                if dialogue_log_status == ToolInvokeMessage.LogMessage.LogStatus.SUCCESS:
                     self.session.storage.delete(params.storage_key)

            # --- Final Steps for the Round ---
            yield self.create_json_message({ # Yield metadata
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: self.llm_usage["usage"].total_price if self.llm_usage["usage"] else 0,
                    LogMetadata.CURRENCY: self.llm_usage["usage"].currency if self.llm_usage["usage"] else "",
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                }
            })

            # Finish the main round log
            yield self.finish_log_message(
                log=round_log_msg, # Correlate with the START round message
                data={
                    "output": final_output_message if final_output_message else "No text output generated this turn.",
                    "final_context": final_context_state,
                    "status": final_status,
                    "llm_result": result
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                },
                status=dialogue_log_status # Round status reflects inner processing status
            )


        except Exception as e:
             # --- Catch unexpected errors ---
             current_time = time.perf_counter()
             start_time_for_meta = llm_start_time if 'llm_start_time' in locals() else round_started_at
             dialogue_error_data = {"error": f"Unhandled exception during processing: {str(e)}"}
             if result: dialogue_error_data["response_before_error"] = result

             yield self.finish_log_message(
                 log=dialogue_log_msg,
                 data=dialogue_error_data,
                 metadata={
                     LogMetadata.STARTED_AT: start_time_for_meta,
                     LogMetadata.FINISHED_AT: current_time,
                     LogMetadata.ELAPSED_TIME: current_time - start_time_for_meta,
                 },
                 status=ToolInvokeMessage.LogMessage.LogStatus.ERROR
             )
             error_message = f"Sorry, an internal error occurred while processing your request. Please try again later."
             yield self.create_text_message(error_message)
             yield self.finish_log_message(
                 log=round_log_msg,
                 data={"output": error_message, "status": f"Unhandled Exception: {str(e)}"},
                 metadata={
                     LogMetadata.STARTED_AT: round_started_at,
                     LogMetadata.FINISHED_AT: current_time,
                     LogMetadata.ELAPSED_TIME: current_time - round_started_at,
                     LogMetadata.TOTAL_TOKENS: self.llm_usage["usage"].total_tokens if self.llm_usage["usage"] else 0,
                 },
                 status=ToolInvokeMessage.LogMessage.LogStatus.ERROR
             )
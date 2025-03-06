# Dialogue Agent

## tod_agent

**Author:** [svcvit](https://github.com/svcvit)
**Version:** 0.0.2
**Type:** agent

### Description

A powerful task-oriented dialogue agent that can collect information through structured conversations. It supports dynamic field validation, multi-field information extraction, and state management.

### Features

- Task-oriented dialogue management
- Dynamic field validation
- Multi-field information extraction
- Conversation state persistence
- Automatic answer validation
- Context-aware information collection
- Natural language interaction

### Usage Guide

#### Parameters

1. **task_schema** (Required)
   - Type: string (JSON)
   - Description: Schema defining the fields to collect
   - Example（Please use this code to test it out. There should be a space in front of the `{`）:
     ```json
      {
       "fields": [
         {
           "name": "destination",
           "question": "请问您想去哪里旅行？",
           "required": true
         },
         {
           "name": "duration",
           "question": "您计划旅行多长时间？",
           "required": true
         },
         {
           "name": "budget",
           "question": "您的预算大约是多少？",
           "required": true
         }
       ]
     }
     ```

2. **query** (Required)
   - Type: string
   - Description: User's input text
   - Example: `"我想去日本玩三天"`

3. **model** (Required)
   - Type: AgentModelConfig
   - Description: LLM model configuration
   - Example: Configuration for GPT or other LLM models

4. **storage_key** (Required)
   - Type: string
   - Description: Unique key for storing conversation state
   - Example: `"conversation-123"`

#### Response Format

The agent returns messages in the following formats:

1. Text messages for questions and responses
2. JSON message with collected data when complete
3. Summary message when all fields are collected

### Changelog

#### v0.0.2
- Added logging functionality
- Added token usage statistics
- Optimized code structure and performance

#### v0.0.1
- Project initialization
- Implemented multi-turn dialogue
- Implemented conversation state storage
- Implemented intelligent Q&A content extraction
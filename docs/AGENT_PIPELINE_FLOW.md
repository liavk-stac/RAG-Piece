# One Piece Chatbot - Agent Pipeline Flow

## Overview
This document explains how the One Piece Chatbot processes user queries and images through its intelligent agent pipeline, from initial input to final response.

## Pipeline Architecture

The chatbot uses a **7-agent pipeline** that works together to understand, process, and respond to user input:

**Planning Agent:**
- **Router Agent** - Analyzes input and creates execution plan (does not execute in pipeline)

**Execution Pipeline:**
1. **Search Agent** - Retrieves relevant knowledge from RAG database
2. **Image Analysis Agent** - Processes uploaded images (if any)
3. **Image Retrieval Agent** - Finds and selects relevant images from database
4. **Timeline Agent** - Handles temporal and chronological queries (if needed)
5. **Reasoning Agent** - Performs logical analysis and relationship extraction (if needed)
6. **Response Agent** - Synthesizes all information into final response

## Complete Flow: Input → Output

### 1. **Input Reception**
- **Text Query**: User types a question or statement
- **Image Upload**: User uploads an image file (optional)
- **Combined Input**: Both text and image together
- **Session Context**: Previous conversation history is loaded

### 2. **Router Agent Analysis** 🧠
**What it does:**
- Analyzes the user's intent (search, analysis, timeline, etc.)
- Determines query complexity (simple, moderate, complex)
- Identifies input modality (text, image, or multimodal)
- Creates an execution plan specifying which agents to use and in what order

**Example:**
- Query: "Tell me about Luffy's relationship with Shanks"
- Intent: Analysis + Relationship
- Complexity: Moderate
- Modality: Text-only
- Plan: Router → Search → Image Retrieval → Reasoning → Response

### 3. **Agent Execution Pipeline** ⚙️
**Agents execute based on the router's plan:**

#### **Search Agent** (if needed)
- Enhances the query using LLM intelligence
- Searches the One Piece RAG database using hybrid search (BM25 + FAISS)
- Retrieves relevant knowledge chunks with confidence scores
- Provides context and background information

#### **Image Analysis Agent** (if image uploaded)
- Validates image format, size, and quality
- Uses GPT-4o vision model to generate detailed descriptions
- Cross-references image content with One Piece knowledge
- Provides comprehensive visual analysis with confidence scores

#### **Image Retrieval Agent** (if relevant images needed)
- Intelligently selects relevant images from the image database
- Uses LLM to analyze user queries for image-related intent
- Matches query intent with image metadata (character, location, scene)
- Selects the most relevant single image with relevance scoring
- Provides image data for frontend display alongside text responses

#### **Reasoning Agent** (if needed)
- Analyzes search results for logical patterns
- Identifies causal connections and relationships
- Extracts comparative elements and classifications
- Generates inferences based on available information

#### **Timeline Agent** (if temporal query)
- Analyzes chronological aspects of the query
- Identifies One Piece eras and temporal context
- Provides timeline relationship analysis
- Handles "when" and "chronological" questions

### 4. **Response Synthesis** 🎯
**Response Agent combines all agent outputs:**
- Merges information from all relevant agents
- Applies response templates based on query type
- Calculates overall confidence score
- Formats response with sources and metadata
- Includes relevant images if retrieved

### 5. **Output Delivery** 📤
**Final response includes:**
- **Text Response**: Comprehensive answer to user's query
- **Confidence Score**: How certain the system is about the response
- **Processing Time**: How long the analysis took
- **Image Data**: Relevant images retrieved from database (if any)
- **Metadata**: Which agents were used, sources consulted
- **Session Info**: Conversation tracking and context

## Key Features

### **Intelligent Routing**
- Automatically determines which agents are needed
- Skips unnecessary agents for simple queries
- Adapts execution plan based on query complexity
- **Execution Order**: Search → Image Analysis → Image Retrieval → Timeline → Reasoning → Response

### **Context Awareness**
- Remembers previous conversation turns
- Uses conversation history to improve responses
- Maintains session context across interactions

### **Multimodal Processing**
- Handles text queries independently
- Processes images through vision models
- Combines text and image analysis when both provided

### **Performance Optimization**
- Configurable timeouts for each agent (30 seconds)
- Retry logic for failed operations (3 attempts)
- Response caching for repeated queries (5 minutes)

## Example Workflow

**User Input:** "Show me a picture of the Straw Hat Pirates and tell me about their journey"

**Pipeline Flow:**
1. **Router**: Identifies as complex, multimodal query
2. **Search**: Retrieves information about Straw Hat Pirates' journey
3. **Image Analysis**: Not needed (no image uploaded)
4. **Image Retrieval**: Finds relevant crew image from database
5. **Reasoning**: Analyzes journey patterns and crew dynamics
6. **Response**: Combines text response with retrieved image

**Output:**
- Comprehensive text about the crew's journey
- Relevant crew image displayed
- Confidence score and processing metadata
- Session maintained for follow-up questions

## Error Handling

- **Agent Failures**: Individual agent timeouts and retries
- **API Errors**: Graceful degradation with user feedback
- **Invalid Input**: Clear error messages and suggestions
- **System Issues**: Robust error recovery and logging

## Performance Characteristics

- **Typical Response Time**: 5-15 seconds for complex queries
- **Simple Queries**: 2-5 seconds
- **Image Analysis**: 8-20 seconds (depending on complexity)
- **Memory Usage**: Efficient session management with configurable retention

---

*This pipeline ensures intelligent, context-aware responses while maintaining performance and reliability across all types of One Piece queries.*

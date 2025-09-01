"""
Pipeline Logger for One Piece Chatbot

Comprehensive logging utility that captures the entire agent pipeline execution
in a single log file for easy debugging and analysis.
"""

import logging
import logging.handlers
import os
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback


@dataclass
class PipelineLogEntry:
    """Structured log entry for pipeline operations."""
    timestamp: str
    level: str
    agent: str
    operation: str
    message: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error_details: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None


class PipelineLogger:
    """
    Comprehensive logger for the One Piece Chatbot agent pipeline.
    
    Captures all agent inputs, outputs, LLM calls, and execution flow
    in a single structured log file for easy debugging and analysis.
    """
    
    def __init__(self, config):
        """Initialize the pipeline logger."""
        self.config = config
        self.logger = self._setup_logger()
        self.current_session_id = None
        self.current_query = None
        self.pipeline_start_time = None
        
    def _setup_logger(self) -> logging.Logger:
        """Set up the main pipeline logger."""
        logger = logging.getLogger("one_piece_pipeline")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter for structured logging
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler (always enabled)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for comprehensive logging
        if self.config.LOG_TO_FILE:
            self._setup_file_handler(logger, formatter)
        
        return logger
    
    def _setup_file_handler(self, logger: logging.Logger, formatter: logging.Formatter):
        """Set up file handler with rotation."""
        # Ensure logs directory exists
        log_dir = os.path.dirname(self.config.LOG_FILE_PATH)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.config.LOG_FILE_PATH,
            maxBytes=self.config.LOG_MAX_SIZE,
            backupCount=self.config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def start_pipeline(self, query: str, session_id: str, modality: str = "text"):
        """Start logging a new pipeline execution."""
        self.current_query = query
        self.current_session_id = session_id
        self.pipeline_start_time = time.time()
        
        self.logger.info(f"[START] PIPELINE START - Query: '{query}' | Session: {session_id} | Modality: {modality}")
        self.logger.info(f"[TIME] Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def end_pipeline(self, success: bool, total_time: float, agent_outputs: Dict[str, Any]):
        """End logging for a pipeline execution."""
        status = "[SUCCESS]" if success else "[FAILED]"
        self.logger.info(f"[END] PIPELINE END - {status} | Total time: {total_time:.2f}s")
        
        if agent_outputs:
            self.logger.info(f"[INFO] Agents executed: {list(agent_outputs.keys())}")
        
        self.logger.info(f"[TIME] Pipeline ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        
        # Reset pipeline state
        self.current_session_id = None
        self.current_query = None
        self.pipeline_start_time = None
    
    def log_agent_start(self, agent_name: str, operation: str, input_data: Optional[Dict[str, Any]] = None):
        """Log the start of an agent execution."""
        if not self.config.ENABLE_PIPELINE_LOGGING:
            return
            
        self.logger.info(f"[AGENT] AGENT START - {agent_name.upper()} | Operation: {operation}")
        
        if input_data:
            # Log complete input data (sanitized for sensitive information)
            sanitized_input = self._sanitize_input_data(input_data)
            self.logger.info(f"[INPUT] {agent_name.upper()} COMPLETE INPUT DATA:")
            self.logger.info(json.dumps(sanitized_input, indent=2, ensure_ascii=False))
    
    def log_agent_end(self, agent_name: str, operation: str, output_data: Optional[Dict[str, Any]] = None, 
                      execution_time: float = 0.0, success: bool = True):
        """Log the end of an agent execution."""
        if not self.config.ENABLE_PIPELINE_LOGGING:
            return
            
        status = "[SUCCESS]" if success else "[FAILED]"
        self.logger.info(f"[AGENT] AGENT END - {agent_name.upper()} | {status} | Time: {execution_time:.2f}s")
        
        if output_data:
            # Log complete output data (sanitized)
            sanitized_output = self._sanitize_output_data(output_data)
            self.logger.info(f"[OUTPUT] {agent_name.upper()} COMPLETE OUTPUT DATA:")
            self.logger.info(json.dumps(sanitized_output, indent=2, ensure_ascii=False))
    
    def log_llm_call(self, agent_name: str, operation: str, prompt: str, response: str, 
                     model: str, tokens_used: Optional[int] = None, execution_time: float = 0.0):
        """Log an LLM API call with full details."""
        if not self.config.ENABLE_PIPELINE_LOGGING:
            return
        self.logger.info(f"[LLM] LLM CALL - {agent_name.upper()} | {operation} | Model: {model} | Time: {execution_time:.2f}s")
        
        if self.config.ENABLE_LLM_CALL_LOGGING:
            self.logger.info(f"[PROMPT] LLM COMPLETE PROMPT ({agent_name.upper()}):")
            self.logger.info(prompt)
            self.logger.info(f"[RESPONSE] LLM COMPLETE RESPONSE ({agent_name.upper()}):")
            self.logger.info(response)
            
            if tokens_used:
                self.logger.info(f"[TOKENS] Tokens used: {tokens_used}")
    
    def log_agent_error(self, agent_name: str, operation: str, error: Exception, 
                        input_data: Optional[Dict[str, Any]] = None):
        """Log an agent execution error."""
        self.logger.error(f"[ERROR] AGENT ERROR - {agent_name.upper()} | {operation}")
        self.logger.error(f"[ERROR] Error: {str(error)}")
        self.logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        
        if input_data:
            sanitized_input = self._sanitize_input_data(input_data)
            self.logger.error(f"[INPUT] Input data: {json.dumps(sanitized_input, indent=2)}")
    
    def log_pipeline_decision(self, decision_type: str, details: Dict[str, Any]):
        """Log pipeline routing and decision information."""
        self.logger.info(f"[DECISION] PIPELINE DECISION - {decision_type.upper()}")
        self.logger.debug(f"[DETAILS] Decision details: {json.dumps(details, indent=2)}")
    
    def log_memory_operation(self, operation: str, details: Dict[str, Any]):
        """Log memory and session operations."""
        self.logger.debug(f"[MEMORY] MEMORY - {operation}: {json.dumps(details, indent=2)}")
    
    def log_performance_metric(self, metric_name: str, value: Any, unit: str = ""):
        """Log performance metrics."""
        self.logger.debug(f"[PERF] PERFORMANCE - {metric_name}: {value}{unit}")
    
    def log_llm_call(self, agent_name: str, prompt: str, response: str, tokens_used: int, system_message: str = ""):
        """Log complete LLM API call details."""
        if not self.config.ENABLE_PIPELINE_LOGGING:
            return
        self.logger.info(f"[LLM] {agent_name.upper()} LLM CALL:")
        
        if system_message:
            self.logger.info(f"[LLM] {agent_name.upper()} SYSTEM MESSAGE:")
            self.logger.info(system_message)
        
        self.logger.info(f"[LLM] {agent_name.upper()} COMPLETE PROMPT:")
        self.logger.info(prompt)
        
        if response:
            self.logger.info(f"[LLM] {agent_name.upper()} COMPLETE RESPONSE:")
            self.logger.info(response)
            self.logger.info(f"[LLM] {agent_name.upper()} TOKENS USED: {tokens_used}")
        
        self.logger.info("-" * 80)

    def log_image_analysis(self, image_data: Dict[str, Any], analysis_result: Dict[str, Any]):
        """Log complete image analysis details."""
        self.logger.info(f"[IMAGE] IMAGE ANALYSIS COMPLETE DATA:")
        self.logger.info(f"[IMAGE] Image Metadata: {json.dumps(image_data, indent=2, ensure_ascii=False)}")
        self.logger.info(f"[IMAGE] Analysis Result: {json.dumps(analysis_result, indent=2, ensure_ascii=False)}")
    
    def log_rag_retrieval(self, query: str, retrieved_chunks: List[Dict[str, Any]], scores: List[float], search_metadata: Optional[Dict[str, Any]] = None):
        """Log RAG database retrieval details with complete metadata."""
        self.logger.info(f"[RAG] RAG RETRIEVAL COMPLETE DATA:")
        self.logger.info(f"[RAG] Original Query: {query}")
        
        # Log search metadata if available
        if search_metadata:
            self.logger.info(f"[RAG] Search Metadata:")
            self.logger.info(json.dumps(search_metadata, indent=2, ensure_ascii=False))
            
            # Extract and log specific search details
            search_params = search_metadata.get('search_parameters', {})
            if search_params:
                self.logger.info(f"[RAG] Enhanced Query: {search_params.get('enhanced_query', 'N/A')}")
                self.logger.info(f"[RAG] Search Strategy: {search_params.get('strategy', 'N/A')}")
                self.logger.info(f"[RAG] One Piece Terms: {search_params.get('one_piece_terms', [])}")
                self.logger.info(f"[RAG] Search Limit: {search_params.get('search_limit', 'N/A')}")
                self.logger.info(f"[RAG] LLM Enhanced: {search_params.get('llm_enhanced', 'N/A')}")
            
            # Log search performance details
            search_perf = search_metadata.get('search_performance', {})
            if search_perf:
                self.logger.info(f"[RAG] Search Performance:")
                self.logger.info(json.dumps(search_perf, indent=2, ensure_ascii=False))
        
        self.logger.info(f"[RAG] Retrieved Chunks ({len(retrieved_chunks)}):")
        
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores)):
            self.logger.info(f"[RAG] Chunk {i+1} (Score: {score:.4f}):")
            
            # Extract and log complete metadata structure
            metadata = chunk.get('metadata', {})
            search_metadata = chunk.get('search_metadata', {})
            content = chunk.get('content', '')
            source = chunk.get('source', '')
            chunk_id = chunk.get('chunk_id', '')
            
            # Calculate token count (approximate: words * 1.3)
            word_count = len(content.split()) if content else 0
            token_count = int(word_count * 1.3) if word_count > 0 else 0
            
            # Log basic chunk info
            chunk_info = {
                'chunk_id': chunk_id,
                'source': source,
                'content_preview': content[:200] + "..." if len(content) > 200 else content,
                'content_length': len(content),
                'token_count': token_count,
                'relevance_score': score
            }
            self.logger.info(f"[RAG] Chunk {i+1} Basic Info:")
            self.logger.info(json.dumps(chunk_info, indent=2, ensure_ascii=False))
            
            # Log all requested metadata fields (show all fields, even if empty)
            useful_metadata = {}
            
            # Try search_metadata first, then fallback to metadata
            meta_source = search_metadata if search_metadata else metadata
            
            # Always show these fields, even if empty/None
            useful_metadata['chunk_id'] = chunk_id or 'Empty'
            useful_metadata['article_name'] = meta_source.get('article_name', 'Empty') or 'Empty'
            useful_metadata['sub_article_name'] = meta_source.get('sub_article_name', 'Empty') or 'Empty'
            useful_metadata['section_name'] = meta_source.get('section_name', 'Empty') or 'Empty'
            useful_metadata['subsection_names'] = meta_source.get('subsection_names', []) or []
            useful_metadata['keywords'] = meta_source.get('keywords', []) or []
            
            self.logger.info(f"[RAG] Chunk {i+1} Metadata:")
            self.logger.info(json.dumps(useful_metadata, indent=2, ensure_ascii=False))
            
            # Log full chunk content if not too long
            if len(content) <= 1000:
                self.logger.info(f"[RAG] Chunk {i+1} Full Content:")
                self.logger.info(content)
            else:
                self.logger.info(f"[RAG] Chunk {i+1} Content (truncated - first 500 chars):")
                self.logger.info(content[:500] + "...")
                self.logger.info(f"[RAG] Chunk {i+1} Content (truncated - last 500 chars):")
                self.logger.info("..." + content[-500:])
            
            self.logger.info(f"[RAG] Chunk {i+1} End")
            self.logger.info("-" * 60)
    
    def log_image_retrieval(self, query: str, selected_images: List[Dict[str, Any]], reasoning: str):
        """Log image retrieval details with file paths."""
        self.logger.info(f"[IMAGE_RETRIEVAL] IMAGE RETRIEVAL COMPLETE DATA:")
        self.logger.info(f"[IMAGE_RETRIEVAL] Query: {query}")
        self.logger.info(f"[IMAGE_RETRIEVAL] Selection Reasoning: {reasoning}")
        self.logger.info(f"[IMAGE_RETRIEVAL] Selected Images ({len(selected_images)}):")
        for i, image in enumerate(selected_images):
            self.logger.info(f"[IMAGE_RETRIEVAL] Image {i+1}:")
            self.logger.info(json.dumps(image, indent=2, ensure_ascii=False))
    
    def log_pipeline_output(self, final_response: Dict[str, Any], all_agent_outputs: Dict[str, Any]):
        """Log complete pipeline final output."""
        if not self.config.ENABLE_PIPELINE_LOGGING:
            return
            
        self.logger.info(f"[PIPELINE] COMPLETE PIPELINE OUTPUT:")
        self.logger.info(f"[PIPELINE] Final Response: {json.dumps(final_response, indent=2, ensure_ascii=False)}")
        self.logger.info(f"[PIPELINE] All Agent Outputs: {json.dumps(all_agent_outputs, indent=2, ensure_ascii=False)}")
    
    def _sanitize_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data for logging (remove sensitive information)."""
        sanitized = data.copy()
        
        # Remove potentially sensitive fields
        sensitive_fields = ['api_key', 'password', 'token', 'secret']
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = '[REDACTED]'
        
        # Limit large text fields
        if 'query' in sanitized and len(str(sanitized['query'])) > 500:
            sanitized['query'] = str(sanitized['query'])[:500] + "...[TRUNCATED]"
        
        return sanitized
    
    def _sanitize_output_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output data for logging."""
        sanitized = data.copy()
        
        # Limit large response fields
        if 'response' in sanitized and len(str(sanitized['response'])) > 1000:
            sanitized['response'] = str(sanitized['response'])[:1000] + "...[TRUNCATED]"
        
        return sanitized
    
    def log_separator(self, message: str = ""):
        """Log a visual separator in the log."""
        if message:
            self.logger.info(f"--- {message} ---")
        else:
            self.logger.info("-" * 50)


# Global pipeline logger instance
_pipeline_logger = None

def get_pipeline_logger(config) -> PipelineLogger:
    """Get or create the global pipeline logger instance."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = PipelineLogger(config)
    return _pipeline_logger

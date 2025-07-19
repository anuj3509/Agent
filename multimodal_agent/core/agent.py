"""
Main Multimodal AI Agent using LangGraph for content creation and analysis
"""
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import BaseTool

from ..config.settings import settings
from ..utils.logger import get_logger
from ..modules.video_captions.processor import VideoCaptionProcessor
from ..modules.multimedia_narratives.generator import NarrativeGenerator
from ..modules.marketing_content.creator import MarketingContentCreator

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State for the multimodal agent"""
    messages: List[BaseMessage]
    task_type: str
    input_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    output: Dict[str, Any]
    error: Optional[str]


class MultimodalAgent:
    """
    Main multimodal AI agent that orchestrates content creation and analysis
    using LangGraph for workflow management.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the multimodal agent"""
        self.api_key = openai_api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=settings.default_text_model,
            temperature=0.7
        )
        
        # Initialize vision model
        self.vision_llm = ChatOpenAI(
            api_key=self.api_key,
            model=settings.default_vision_model,
            temperature=0.7
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=settings.default_embedding_model
        )
        
        # Initialize processors
        self.video_processor = VideoCaptionProcessor(self.vision_llm)
        self.narrative_generator = NarrativeGenerator(self.llm, self.embeddings)
        self.marketing_creator = MarketingContentCreator(self.llm, self.vision_llm)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        logger.info("MultimodalAgent initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the agent"""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_task", self._route_task)
        workflow.add_node("process_video_captions", self._process_video_captions)
        workflow.add_node("generate_narrative", self._generate_narrative)
        workflow.add_node("create_marketing_content", self._create_marketing_content)
        workflow.add_node("combine_outputs", self._combine_outputs)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the entry point
        workflow.set_entry_point("route_task")
        
        # Add conditional edges from routing
        workflow.add_conditional_edges(
            "route_task",
            self._decide_next_step,
            {
                "video_captions": "process_video_captions",
                "multimedia_narrative": "generate_narrative", 
                "marketing_content": "create_marketing_content",
                "error": "handle_error"
            }
        )
        
        # Add edges to combine outputs
        workflow.add_edge("process_video_captions", "combine_outputs")
        workflow.add_edge("generate_narrative", "combine_outputs")
        workflow.add_edge("create_marketing_content", "combine_outputs")
        
        # Add terminal edges
        workflow.add_edge("combine_outputs", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _route_task(self, state: AgentState) -> AgentState:
        """Route the task based on input analysis"""
        try:
            input_data = state["input_data"]
            task_type = state.get("task_type", "")
            
            if not task_type:
                # Auto-detect task type based on input
                if "video" in input_data or "video_path" in input_data:
                    task_type = "video_captions"
                elif "narrative" in input_data or "story" in input_data:
                    task_type = "multimedia_narrative"
                elif "marketing" in input_data or "campaign" in input_data:
                    task_type = "marketing_content"
                else:
                    # Default to narrative if unclear
                    task_type = "multimedia_narrative"
            
            state["task_type"] = task_type
            logger.info(f"Task routed to: {task_type}")
            
        except Exception as e:
            logger.error(f"Error in task routing: {e}")
            state["error"] = str(e)
            state["task_type"] = "error"
        
        return state
    
    def _decide_next_step(self, state: AgentState) -> str:
        """Decide the next step based on task type"""
        task_type = state.get("task_type", "error")
        return task_type if task_type != "error" else "error"
    
    async def _process_video_captions(self, state: AgentState) -> AgentState:
        """Process video caption generation"""
        try:
            input_data = state["input_data"]
            video_path = input_data.get("video_path")
            
            if not video_path:
                raise ValueError("Video path is required for caption generation")
            
            # Process video captions
            captions = await self.video_processor.generate_captions(
                video_path=Path(video_path),
                context=input_data.get("context", "")
            )
            
            state["processed_data"] = {"captions": captions}
            logger.info("Video captions generated successfully")
            
        except Exception as e:
            logger.error(f"Error in video caption processing: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _generate_narrative(self, state: AgentState) -> AgentState:
        """Generate multimedia narrative"""
        try:
            input_data = state["input_data"]
            
            narrative = await self.narrative_generator.create_narrative(
                topic=input_data.get("topic", ""),
                media_elements=input_data.get("media_elements", []),
                style=input_data.get("style", "engaging"),
                target_audience=input_data.get("target_audience", "general")
            )
            
            state["processed_data"] = {"narrative": narrative}
            logger.info("Multimedia narrative generated successfully")
            
        except Exception as e:
            logger.error(f"Error in narrative generation: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _create_marketing_content(self, state: AgentState) -> AgentState:
        """Create targeted marketing content"""
        try:
            input_data = state["input_data"]
            
            marketing_content = await self.marketing_creator.create_content(
                product_info=input_data.get("product_info", {}),
                target_audience=input_data.get("target_audience", ""),
                content_type=input_data.get("content_type", "social_media_post"),
                visual_elements=input_data.get("visual_elements", [])
            )
            
            state["processed_data"] = {"marketing_content": marketing_content}
            logger.info("Marketing content created successfully")
            
        except Exception as e:
            logger.error(f"Error in marketing content creation: {e}")
            state["error"] = str(e)
        
        return state
    
    def _combine_outputs(self, state: AgentState) -> AgentState:
        """Combine and format final outputs"""
        try:
            processed_data = state.get("processed_data", {})
            task_type = state.get("task_type", "")
            
            output = {
                "task_type": task_type,
                "status": "success",
                "data": processed_data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            state["output"] = output
            logger.info("Outputs combined successfully")
            
        except Exception as e:
            logger.error(f"Error combining outputs: {e}")
            state["error"] = str(e)
        
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error occurred")
        
        output = {
            "status": "error",
            "error": error,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        state["output"] = output
        logger.error(f"Workflow error handled: {error}")
        
        return state
    
    async def process(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a multimodal content creation task
        
        Args:
            task_type: Type of task ("video_captions", "multimedia_narrative", "marketing_content")
            input_data: Input data for the task
            
        Returns:
            Dictionary containing the results
        """
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=f"Process {task_type} task")],
            task_type=task_type,
            input_data=input_data,
            processed_data={},
            output={},
            error=None
        )
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            return result["output"]
            
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def generate_video_captions(self, video_path: str, context: str = "") -> Dict[str, Any]:
        """Generate captions for a video"""
        return await self.process("video_captions", {
            "video_path": video_path,
            "context": context
        })
    
    async def create_multimedia_narrative(self, topic: str, media_elements: List[str], 
                                        style: str = "engaging", target_audience: str = "general") -> Dict[str, Any]:
        """Create an interactive multimedia narrative"""
        return await self.process("multimedia_narrative", {
            "topic": topic,
            "media_elements": media_elements,
            "style": style,
            "target_audience": target_audience
        })
    
    async def create_marketing_content(self, product_info: Dict[str, Any], target_audience: str,
                                     content_type: str = "social_media_post", 
                                     visual_elements: List[str] = None) -> Dict[str, Any]:
        """Create targeted marketing content"""
        return await self.process("marketing_content", {
            "product_info": product_info,
            "target_audience": target_audience,
            "content_type": content_type,
            "visual_elements": visual_elements or []
        }) 
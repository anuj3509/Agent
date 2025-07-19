"""
Multimedia narrative generation using LangChain and embeddings
"""
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MediaElement:
    """Represents a media element in the narrative"""
    type: str  # "image", "video", "audio", "text"
    content: str  # path or content
    description: str
    timestamp: Optional[float] = None
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NarrativeSegment:
    """Represents a segment of the narrative"""
    id: str
    content: str
    media_elements: List[MediaElement]
    interactions: List[Dict[str, Any]]
    transitions: List[str]
    metadata: Dict[str, Any]


class NarrativeGenerator:
    """
    Generates interactive multimedia narratives using AI
    """
    
    def __init__(self, llm: ChatOpenAI, embeddings: OpenAIEmbeddings):
        """
        Initialize the narrative generator
        
        Args:
            llm: Language model for text generation
            embeddings: Embeddings model for semantic search
        """
        self.llm = llm
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Narrative templates and styles
        self.narrative_styles = {
            "engaging": "Create an engaging, conversational narrative with vivid descriptions",
            "educational": "Focus on educational content with clear explanations and learning objectives",
            "dramatic": "Use dramatic storytelling techniques with tension and emotion",
            "documentary": "Present information in a factual, documentary-style format",
            "interactive": "Design for user interaction with choices and branching paths"
        }
        
        self.content_templates = {
            "hero_journey": self._get_hero_journey_template(),
            "problem_solution": self._get_problem_solution_template(),
            "chronological": self._get_chronological_template(),
            "compare_contrast": self._get_compare_contrast_template(),
            "cause_effect": self._get_cause_effect_template()
        }
    
    async def create_narrative(self, topic: str, media_elements: List[str],
                             style: str = "engaging", target_audience: str = "general",
                             narrative_type: str = "hero_journey",
                             max_segments: int = 10) -> Dict[str, Any]:
        """
        Create a multimedia narrative
        
        Args:
            topic: Main topic of the narrative
            media_elements: List of media element descriptions or paths
            style: Narrative style
            target_audience: Target audience description
            narrative_type: Type of narrative structure
            max_segments: Maximum number of segments
            
        Returns:
            Dictionary containing the complete narrative
        """
        try:
            logger.info(f"Creating narrative: topic='{topic}', style='{style}', type='{narrative_type}'")
            
            # Process media elements
            processed_media = await self._process_media_elements(media_elements)
            
            # Generate narrative outline
            outline = await self._generate_narrative_outline(
                topic, processed_media, style, target_audience, narrative_type
            )
            
            # Create detailed segments
            segments = await self._create_narrative_segments(
                outline, processed_media, style, target_audience, max_segments
            )
            
            # Generate interactive elements
            interactions = await self._generate_interactions(segments, target_audience)
            
            # Create final narrative structure
            narrative = {
                "title": outline.get("title", topic),
                "description": outline.get("description", ""),
                "topic": topic,
                "style": style,
                "target_audience": target_audience,
                "narrative_type": narrative_type,
                "segments": segments,
                "interactions": interactions,
                "media_elements": processed_media,
                "metadata": {
                    "total_segments": len(segments),
                    "estimated_duration": self._estimate_duration(segments),
                    "complexity_level": self._assess_complexity(segments),
                    "interactive_points": len(interactions)
                }
            }
            
            logger.info(f"Narrative created successfully: {len(segments)} segments, {len(interactions)} interactions")
            return narrative
            
        except Exception as e:
            logger.error(f"Error creating narrative: {e}")
            raise
    
    async def _process_media_elements(self, media_elements: List[str]) -> List[MediaElement]:
        """
        Process and categorize media elements
        
        Args:
            media_elements: Raw media element descriptions
            
        Returns:
            List of processed MediaElement objects
        """
        processed_elements = []
        
        for i, element in enumerate(media_elements):
            try:
                # Determine media type
                media_type = self._determine_media_type(element)
                
                # Generate description if it's a path
                if Path(element).exists():
                    description = await self._generate_media_description(element, media_type)
                else:
                    description = element
                
                # Create MediaElement
                media_element = MediaElement(
                    type=media_type,
                    content=element,
                    description=description,
                    metadata={"index": i, "processed": True}
                )
                
                processed_elements.append(media_element)
                
            except Exception as e:
                logger.warning(f"Error processing media element '{element}': {e}")
                # Create a text element as fallback
                processed_elements.append(MediaElement(
                    type="text",
                    content=element,
                    description=element,
                    metadata={"index": i, "processed": False, "error": str(e)}
                ))
        
        logger.info(f"Processed {len(processed_elements)} media elements")
        return processed_elements
    
    async def _generate_narrative_outline(self, topic: str, media_elements: List[MediaElement],
                                        style: str, target_audience: str, narrative_type: str) -> Dict[str, Any]:
        """
        Generate the overall narrative outline
        """
        # Get template and style description
        template = self.content_templates.get(narrative_type, self.content_templates["hero_journey"])
        style_description = self.narrative_styles.get(style, self.narrative_styles["engaging"])
        
        # Create media summary
        media_summary = "\n".join([
            f"- {elem.type.upper()}: {elem.description[:100]}..."
            for elem in media_elements[:10]  # Limit for prompt size
        ])
        
        prompt = f"""
        Create a narrative outline for a multimedia story with the following specifications:
        
        Topic: {topic}
        Style: {style_description}
        Target Audience: {target_audience}
        Narrative Structure: {narrative_type}
        
        Available Media Elements:
        {media_summary}
        
        Template Guidelines:
        {template}
        
        Please provide a JSON outline with:
        1. title: Compelling title for the narrative
        2. description: Brief description of the story
        3. segments: List of 5-10 segment titles with brief descriptions
        4. themes: Key themes to explore
        5. progression: How the story should progress
        6. media_integration: How to integrate the available media
        
        Format as valid JSON.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            outline = json.loads(response.content.strip())
            logger.info("Generated narrative outline successfully")
            return outline
        except json.JSONDecodeError:
            logger.warning("Failed to parse outline JSON, using fallback")
            return self._create_fallback_outline(topic, narrative_type)
    
    async def _create_narrative_segments(self, outline: Dict[str, Any], media_elements: List[MediaElement],
                                       style: str, target_audience: str, max_segments: int) -> List[NarrativeSegment]:
        """
        Create detailed narrative segments
        """
        segments = []
        segment_outlines = outline.get("segments", [])[:max_segments]
        
        for i, segment_info in enumerate(segment_outlines):
            try:
                # Select relevant media for this segment
                segment_media = self._select_media_for_segment(media_elements, i, len(segment_outlines))
                
                # Generate segment content
                content = await self._generate_segment_content(
                    segment_info, media_elements, style, target_audience, i, len(segment_outlines)
                )
                
                # Create transitions
                transitions = self._generate_transitions(i, len(segment_outlines))
                
                # Create segment
                segment = NarrativeSegment(
                    id=f"segment_{i+1}",
                    content=content,
                    media_elements=segment_media,
                    interactions=[],  # Will be populated later
                    transitions=transitions,
                    metadata={
                        "index": i,
                        "title": segment_info.get("title", f"Segment {i+1}"),
                        "description": segment_info.get("description", ""),
                        "estimated_reading_time": len(content.split()) / 200  # 200 WPM
                    }
                )
                
                segments.append(segment)
                
            except Exception as e:
                logger.error(f"Error creating segment {i+1}: {e}")
                # Create fallback segment
                segments.append(self._create_fallback_segment(i, segment_info))
        
        logger.info(f"Created {len(segments)} narrative segments")
        return segments
    
    async def _generate_segment_content(self, segment_info: Dict[str, Any], media_elements: List[MediaElement],
                                      style: str, target_audience: str, segment_index: int, total_segments: int) -> str:
        """
        Generate content for a specific segment
        """
        # Create context about media elements
        media_context = "\n".join([
            f"- {elem.description}" for elem in media_elements
        ])
        
        style_description = self.narrative_styles.get(style, self.narrative_styles["engaging"])
        
        prompt = f"""
        Write content for segment {segment_index + 1} of {total_segments} in a multimedia narrative.
        
        Segment Information:
        Title: {segment_info.get('title', f'Segment {segment_index + 1}')}
        Description: {segment_info.get('description', '')}
        
        Style: {style_description}
        Target Audience: {target_audience}
        
        Available Media Context:
        {media_context}
        
        Requirements:
        1. Write 200-500 words of engaging content
        2. Include natural integration points for multimedia elements
        3. Maintain consistency with the overall narrative
        4. Use appropriate tone for the target audience
        5. Include sensory details and vivid descriptions
        6. Create natural breakpoints for interactions
        
        Format: Plain text with [MEDIA_CUE: description] markers where media should be integrated.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    async def _generate_interactions(self, segments: List[NarrativeSegment], target_audience: str) -> List[Dict[str, Any]]:
        """
        Generate interactive elements for the narrative
        """
        interactions = []
        
        for i, segment in enumerate(segments):
            try:
                # Create interaction prompt
                prompt = f"""
                Design 2-3 interactive elements for a multimedia narrative segment.
                
                Segment Content: {segment.content[:300]}...
                Target Audience: {target_audience}
                Segment Position: {i+1} of {len(segments)}
                
                Create interactions such as:
                1. Multiple choice questions
                2. Interactive hotspots
                3. Branching story paths
                4. User input prompts
                5. Reflection questions
                
                Format as JSON array with each interaction having:
                - type: interaction type
                - content: question or prompt
                - options: available choices (if applicable)
                - feedback: response to user action
                - points: where in the segment to place it
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                
                try:
                    segment_interactions = json.loads(response.content.strip())
                    for interaction in segment_interactions:
                        interaction["segment_id"] = segment.id
                        interactions.append(interaction)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse interactions for segment {i+1}")
                    
            except Exception as e:
                logger.error(f"Error generating interactions for segment {i+1}: {e}")
        
        logger.info(f"Generated {len(interactions)} interactive elements")
        return interactions
    
    def _determine_media_type(self, element: str) -> str:
        """Determine the type of media element"""
        if Path(element).exists():
            ext = Path(element).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                return "image"
            elif ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
                return "video"
            elif ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                return "audio"
            else:
                return "file"
        else:
            # Analyze content to determine type
            if any(word in element.lower() for word in ['image', 'photo', 'picture', 'visual']):
                return "image"
            elif any(word in element.lower() for word in ['video', 'clip', 'movie', 'footage']):
                return "video"
            elif any(word in element.lower() for word in ['audio', 'sound', 'music', 'voice']):
                return "audio"
            else:
                return "text"
    
    async def _generate_media_description(self, media_path: str, media_type: str) -> str:
        """Generate description for media file"""
        # This would integrate with vision/audio models to analyze the media
        # For now, return a placeholder based on file info
        path = Path(media_path)
        return f"{media_type.title()} file: {path.name}"
    
    def _select_media_for_segment(self, media_elements: List[MediaElement], 
                                segment_index: int, total_segments: int) -> List[MediaElement]:
        """Select appropriate media elements for a segment"""
        # Simple distribution strategy - can be enhanced with semantic matching
        elements_per_segment = max(1, len(media_elements) // total_segments)
        start_idx = segment_index * elements_per_segment
        end_idx = min(start_idx + elements_per_segment, len(media_elements))
        
        return media_elements[start_idx:end_idx]
    
    def _generate_transitions(self, segment_index: int, total_segments: int) -> List[str]:
        """Generate transition options for navigation"""
        transitions = []
        
        if segment_index > 0:
            transitions.append("previous")
        if segment_index < total_segments - 1:
            transitions.append("next")
        
        transitions.extend(["menu", "replay"])
        return transitions
    
    def _estimate_duration(self, segments: List[NarrativeSegment]) -> float:
        """Estimate total duration of the narrative in minutes"""
        total_words = sum(len(segment.content.split()) for segment in segments)
        reading_time = total_words / 200  # 200 WPM
        
        # Add time for media and interactions
        media_time = len([elem for segment in segments for elem in segment.media_elements]) * 0.5
        interaction_time = sum(len(segment.interactions) for segment in segments) * 0.5
        
        return reading_time + media_time + interaction_time
    
    def _assess_complexity(self, segments: List[NarrativeSegment]) -> str:
        """Assess the complexity level of the narrative"""
        total_words = sum(len(segment.content.split()) for segment in segments)
        avg_sentence_length = total_words / max(1, sum(segment.content.count('.') for segment in segments))
        
        if avg_sentence_length < 15:
            return "Simple"
        elif avg_sentence_length < 25:
            return "Moderate"
        else:
            return "Complex"
    
    def _create_fallback_outline(self, topic: str, narrative_type: str) -> Dict[str, Any]:
        """Create a fallback outline when JSON parsing fails"""
        return {
            "title": f"A Story About {topic}",
            "description": f"An engaging {narrative_type} narrative exploring {topic}",
            "segments": [
                {"title": "Introduction", "description": f"Introducing the world of {topic}"},
                {"title": "Development", "description": "Exploring the key concepts and ideas"},
                {"title": "Climax", "description": "The central challenge or revelation"},
                {"title": "Resolution", "description": "Bringing everything together"},
                {"title": "Conclusion", "description": "Final thoughts and takeaways"}
            ],
            "themes": [topic, "discovery", "growth"],
            "progression": "linear with interactive elements",
            "media_integration": "throughout all segments"
        }
    
    def _create_fallback_segment(self, index: int, segment_info: Dict[str, Any]) -> NarrativeSegment:
        """Create a fallback segment when generation fails"""
        return NarrativeSegment(
            id=f"segment_{index+1}",
            content=f"This is segment {index+1} about {segment_info.get('title', 'the story')}. "
                   f"{segment_info.get('description', 'Content will be developed here.')}",
            media_elements=[],
            interactions=[],
            transitions=self._generate_transitions(index, 5),
            metadata={
                "index": index,
                "title": segment_info.get("title", f"Segment {index+1}"),
                "description": segment_info.get("description", ""),
                "fallback": True
            }
        )
    
    def _get_hero_journey_template(self) -> str:
        """Get hero's journey narrative template"""
        return """
        Follow the classic hero's journey structure:
        1. Ordinary World - Establish the normal state
        2. Call to Adventure - Present the challenge
        3. Meeting the Mentor - Introduce guidance
        4. Crossing the Threshold - Begin the journey
        5. Tests and Trials - Face obstacles
        6. Revelation - Discover truth or solution
        7. Transformation - Character growth
        8. Return - Apply new knowledge
        """
    
    def _get_problem_solution_template(self) -> str:
        """Get problem-solution narrative template"""
        return """
        Structure around problem-solving:
        1. Problem Identification - What needs to be solved?
        2. Investigation - Exploring the problem
        3. Analysis - Understanding causes and effects
        4. Solution Development - Creating solutions
        5. Implementation - Putting solutions into action
        6. Results - Outcomes and lessons learned
        """
    
    def _get_chronological_template(self) -> str:
        """Get chronological narrative template"""
        return """
        Tell the story in time order:
        1. Beginning - Set the scene and context
        2. Early Development - Initial events
        3. Middle Period - Main developments
        4. Later Development - Building complexity
        5. Recent Events - Leading to the present
        6. Current State - Where we are now
        7. Future Outlook - What comes next
        """
    
    def _get_compare_contrast_template(self) -> str:
        """Get compare and contrast template"""
        return """
        Structure around comparisons:
        1. Introduction - Present items to compare
        2. Similarities - What they have in common
        3. Key Differences - How they differ
        4. Advantages/Disadvantages - Pros and cons
        5. Use Cases - When to use each
        6. Synthesis - Bringing insights together
        """
    
    def _get_cause_effect_template(self) -> str:
        """Get cause and effect template"""
        return """
        Focus on relationships:
        1. Initial Conditions - Setting the stage
        2. Primary Causes - Main driving factors
        3. Chain Reaction - How effects create new causes
        4. Multiple Effects - Various outcomes
        5. Long-term Consequences - Extended impacts
        6. Lessons Learned - Understanding patterns
        """ 
"""
Marketing content creation using multimodal AI analysis
"""
import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

import requests
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProductInfo:
    """Product information for marketing content"""
    name: str
    description: str
    category: str
    price: Optional[float] = None
    key_features: List[str] = None
    benefits: List[str] = None
    target_demographics: List[str] = None
    brand_voice: str = "professional"
    competitors: List[str] = None


@dataclass
class VisualElement:
    """Visual element for marketing content"""
    type: str  # "image", "logo", "product_photo", "lifestyle"
    path: str
    description: str
    style: str
    dominant_colors: List[str] = None
    mood: str = None
    objects: List[str] = None


@dataclass
class MarketingContent:
    """Generated marketing content"""
    content_type: str
    title: str
    content: str
    visual_recommendations: List[Dict[str, Any]]
    call_to_action: str
    target_audience: str
    platform_specific: Dict[str, str]
    metadata: Dict[str, Any]


class MarketingContentCreator:
    """
    Creates targeted marketing content by analyzing visual and textual data
    """
    
    def __init__(self, text_llm: ChatOpenAI, vision_llm: ChatOpenAI):
        """
        Initialize the marketing content creator
        
        Args:
            text_llm: Language model for text generation
            vision_llm: Vision-enabled language model
        """
        self.text_llm = text_llm
        self.vision_llm = vision_llm
        
        # Content type templates
        self.content_types = {
            "social_media_post": {
                "max_length": 280,
                "tone": "engaging and conversational",
                "platforms": ["twitter", "facebook", "instagram", "linkedin"]
            },
            "blog_article": {
                "max_length": 2000,
                "tone": "informative and authoritative",
                "platforms": ["website", "medium", "linkedin"]
            },
            "product_description": {
                "max_length": 500,
                "tone": "persuasive and benefit-focused",
                "platforms": ["ecommerce", "catalog", "website"]
            },
            "email_campaign": {
                "max_length": 800,
                "tone": "personal and action-oriented",
                "platforms": ["email", "newsletter"]
            },
            "ad_copy": {
                "max_length": 150,
                "tone": "compelling and urgent",
                "platforms": ["google_ads", "facebook_ads", "display"]
            },
            "press_release": {
                "max_length": 1500,
                "tone": "formal and newsworthy",
                "platforms": ["media", "pr_wire", "website"]
            }
        }
        
        # Platform-specific formatting
        self.platform_formats = {
            "twitter": {"max_chars": 280, "hashtags": True, "mentions": True},
            "facebook": {"max_chars": 500, "hashtags": True, "links": True},
            "instagram": {"max_chars": 300, "hashtags": True, "visual_focus": True},
            "linkedin": {"max_chars": 700, "professional": True, "hashtags": True},
            "email": {"subject_line": True, "personalization": True, "cta_button": True}
        }
    
    async def create_content(self, product_info: Union[Dict[str, Any], ProductInfo],
                           target_audience: str, content_type: str = "social_media_post",
                           visual_elements: List[str] = None,
                           brand_guidelines: Dict[str, Any] = None,
                           competitor_analysis: bool = False) -> MarketingContent:
        """
        Create targeted marketing content
        
        Args:
            product_info: Product information
            target_audience: Target audience description
            content_type: Type of content to create
            visual_elements: List of visual element paths or descriptions
            brand_guidelines: Brand guidelines and voice
            competitor_analysis: Whether to include competitive analysis
            
        Returns:
            MarketingContent object with generated content
        """
        try:
            logger.info(f"Creating {content_type} for target audience: {target_audience}")
            
            # Process product information
            if isinstance(product_info, dict):
                product = self._process_product_info(product_info)
            else:
                product = product_info
            
            # Analyze visual elements
            visual_analysis = await self._analyze_visual_elements(visual_elements or [])
            
            # Generate audience insights
            audience_insights = await self._analyze_target_audience(target_audience, product)
            
            # Perform competitor analysis if requested
            competitive_insights = {}
            if competitor_analysis and product.competitors:
                competitive_insights = await self._analyze_competitors(product.competitors, content_type)
            
            # Generate core content
            core_content = await self._generate_core_content(
                product, audience_insights, content_type, visual_analysis, competitive_insights
            )
            
            # Create platform-specific variations
            platform_variations = await self._create_platform_variations(
                core_content, content_type, brand_guidelines
            )
            
            # Generate visual recommendations
            visual_recommendations = await self._generate_visual_recommendations(
                core_content, visual_analysis, content_type
            )
            
            # Create final marketing content
            marketing_content = MarketingContent(
                content_type=content_type,
                title=core_content["title"],
                content=core_content["main_content"],
                visual_recommendations=visual_recommendations,
                call_to_action=core_content["cta"],
                target_audience=target_audience,
                platform_specific=platform_variations,
                metadata={
                    "product_name": product.name,
                    "brand_voice": product.brand_voice,
                    "content_length": len(core_content["main_content"]),
                    "visual_elements_analyzed": len(visual_analysis),
                    "platforms": list(platform_variations.keys()),
                    "generation_timestamp": asyncio.get_event_loop().time()
                }
            )
            
            logger.info(f"Marketing content created successfully: {content_type}")
            return marketing_content
            
        except Exception as e:
            logger.error(f"Error creating marketing content: {e}")
            raise
    
    async def _analyze_visual_elements(self, visual_elements: List[str]) -> List[VisualElement]:
        """
        Analyze visual elements using computer vision
        
        Args:
            visual_elements: List of image paths or URLs
            
        Returns:
            List of analyzed VisualElement objects
        """
        analyzed_elements = []
        
        for element in visual_elements:
            try:
                # Load and analyze image
                if element.startswith(('http://', 'https://')):
                    image_data = await self._download_image(element)
                else:
                    image_data = self._load_local_image(element)
                
                if image_data:
                    analysis = await self._analyze_single_image(image_data, element)
                    analyzed_elements.append(analysis)
                    
            except Exception as e:
                logger.warning(f"Error analyzing visual element '{element}': {e}")
                # Create fallback element
                analyzed_elements.append(VisualElement(
                    type="unknown",
                    path=element,
                    description=f"Visual element: {Path(element).name if not element.startswith('http') else element}",
                    style="unknown"
                ))
        
        logger.info(f"Analyzed {len(analyzed_elements)} visual elements")
        return analyzed_elements
    
    async def _analyze_single_image(self, image_data: bytes, source_path: str) -> VisualElement:
        """
        Analyze a single image using vision AI
        
        Args:
            image_data: Image data as bytes
            source_path: Source path or URL
            
        Returns:
            VisualElement with analysis results
        """
        try:
            # Convert to base64 for API
            image_b64 = base64.b64encode(image_data).decode()
            
            # Create analysis prompt
            prompt = """
            Analyze this image for marketing purposes. Provide a detailed analysis including:
            
            1. Visual Style: (modern, vintage, minimalist, bold, elegant, etc.)
            2. Dominant Colors: List 3-5 main colors
            3. Mood/Emotion: What feeling does it convey?
            4. Objects/Elements: What's visible in the image?
            5. Marketing Suitability: How could this be used in marketing?
            6. Target Demographics: What audience would this appeal to?
            
            Format as JSON with keys: style, dominant_colors, mood, objects, marketing_notes, target_demographics
            """
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            )
            
            # Get analysis
            response = await self.vision_llm.ainvoke([message])
            
            try:
                analysis_data = json.loads(response.content.strip())
                
                return VisualElement(
                    type="image",
                    path=source_path,
                    description=analysis_data.get("marketing_notes", "Marketing visual"),
                    style=analysis_data.get("style", "unknown"),
                    dominant_colors=analysis_data.get("dominant_colors", []),
                    mood=analysis_data.get("mood", "neutral"),
                    objects=analysis_data.get("objects", [])
                )
                
            except json.JSONDecodeError:
                # Fallback to text analysis
                return VisualElement(
                    type="image",
                    path=source_path,
                    description=response.content.strip()[:200],
                    style="analyzed",
                    mood="unknown"
                )
                
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return VisualElement(
                type="image",
                path=source_path,
                description="Image for marketing use",
                style="unknown"
            )
    
    async def _analyze_target_audience(self, target_audience: str, product: ProductInfo) -> Dict[str, Any]:
        """
        Analyze target audience and generate insights
        
        Args:
            target_audience: Target audience description
            product: Product information
            
        Returns:
            Dictionary with audience insights
        """
        prompt = f"""
        Analyze the target audience for marketing purposes:
        
        Target Audience: {target_audience}
        Product: {product.name} - {product.description}
        Product Category: {product.category}
        Brand Voice: {product.brand_voice}
        
        Provide insights as JSON with:
        1. demographics: Age, gender, location, income level
        2. psychographics: Values, interests, lifestyle
        3. pain_points: What problems does this audience have?
        4. motivations: What drives their purchasing decisions?
        5. communication_style: How should we talk to them?
        6. preferred_channels: Where do they consume content?
        7. messaging_angles: What appeals to them?
        8. buying_triggers: What makes them purchase?
        
        Format as valid JSON.
        """
        
        try:
            response = await self.text_llm.ainvoke([HumanMessage(content=prompt)])
            insights = json.loads(response.content.strip())
            logger.info("Generated audience insights")
            return insights
        except json.JSONDecodeError:
            logger.warning("Failed to parse audience insights, using fallback")
            return self._create_fallback_audience_insights(target_audience)
    
    async def _generate_core_content(self, product: ProductInfo, audience_insights: Dict[str, Any],
                                   content_type: str, visual_analysis: List[VisualElement],
                                   competitive_insights: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate the core marketing content
        
        Args:
            product: Product information
            audience_insights: Target audience analysis
            content_type: Type of content to create
            visual_analysis: Visual elements analysis
            competitive_insights: Competitive analysis
            
        Returns:
            Dictionary with generated content components
        """
        # Get content type specifications
        content_spec = self.content_types.get(content_type, self.content_types["social_media_post"])
        
        # Create visual context
        visual_context = "\n".join([
            f"- {elem.style} {elem.type} with {elem.mood} mood"
            for elem in visual_analysis
        ]) if visual_analysis else "No visual elements provided"
        
        # Create competitive context
        competitive_context = ""
        if competitive_insights:
            competitive_context = f"\nCompetitive Landscape:\n{competitive_insights.get('summary', '')}"
        
        prompt = f"""
        Create compelling marketing content with these specifications:
        
        PRODUCT:
        Name: {product.name}
        Description: {product.description}
        Category: {product.category}
        Key Features: {', '.join(product.key_features or [])}
        Benefits: {', '.join(product.benefits or [])}
        Brand Voice: {product.brand_voice}
        
        TARGET AUDIENCE:
        Demographics: {audience_insights.get('demographics', {})}
        Pain Points: {audience_insights.get('pain_points', [])}
        Motivations: {audience_insights.get('motivations', [])}
        Communication Style: {audience_insights.get('communication_style', 'professional')}
        
        CONTENT TYPE: {content_type}
        Max Length: {content_spec['max_length']} characters
        Tone: {content_spec['tone']}
        
        VISUAL CONTEXT:
        {visual_context}
        {competitive_context}
        
        Generate:
        1. title: Compelling headline/title
        2. main_content: Core marketing message
        3. cta: Strong call-to-action
        4. key_messages: 3-5 key points to emphasize
        5. emotional_hooks: Emotional triggers to use
        
        Requirements:
        - Align with brand voice and audience communication style
        - Incorporate product benefits (not just features)
        - Address audience pain points and motivations
        - Include emotional appeal appropriate for the content type
        - Ensure content fits within length limits
        - Make it platform-appropriate
        
        Format as valid JSON.
        """
        
        try:
            response = await self.text_llm.ainvoke([HumanMessage(content=prompt)])
            content = json.loads(response.content.strip())
            logger.info(f"Generated core content for {content_type}")
            return content
        except json.JSONDecodeError:
            logger.warning("Failed to parse core content, using fallback")
            return self._create_fallback_content(product, content_type)
    
    async def _create_platform_variations(self, core_content: Dict[str, str], content_type: str,
                                        brand_guidelines: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Create platform-specific variations of the content
        
        Args:
            core_content: Core generated content
            content_type: Type of content
            brand_guidelines: Brand guidelines
            
        Returns:
            Dictionary of platform-specific content
        """
        platforms = self.content_types[content_type]["platforms"]
        variations = {}
        
        for platform in platforms:
            try:
                platform_spec = self.platform_formats.get(platform, {})
                
                prompt = f"""
                Adapt this marketing content for {platform}:
                
                Original Content:
                Title: {core_content.get('title', '')}
                Content: {core_content.get('main_content', '')}
                CTA: {core_content.get('cta', '')}
                
                Platform Requirements for {platform}:
                {json.dumps(platform_spec, indent=2)}
                
                Brand Guidelines: {json.dumps(brand_guidelines or {}, indent=2)}
                
                Adapt the content to:
                1. Fit platform character limits
                2. Use platform-appropriate tone and style
                3. Include platform-specific elements (hashtags, mentions, etc.)
                4. Optimize for platform algorithms and user behavior
                5. Maintain core message while platform-optimizing
                
                Return the adapted content as plain text.
                """
                
                response = await self.text_llm.ainvoke([HumanMessage(content=prompt)])
                variations[platform] = response.content.strip()
                
            except Exception as e:
                logger.warning(f"Error creating {platform} variation: {e}")
                variations[platform] = core_content.get("main_content", "")
        
        logger.info(f"Created variations for {len(variations)} platforms")
        return variations
    
    async def _generate_visual_recommendations(self, core_content: Dict[str, str], 
                                             visual_analysis: List[VisualElement],
                                             content_type: str) -> List[Dict[str, Any]]:
        """
        Generate visual recommendations for the marketing content
        
        Args:
            core_content: Generated content
            visual_analysis: Analyzed visual elements
            content_type: Type of content
            
        Returns:
            List of visual recommendations
        """
        # Analyze existing visuals
        existing_visual_summary = ""
        if visual_analysis:
            existing_visual_summary = "\n".join([
                f"- {elem.style} {elem.type}: {elem.description}"
                for elem in visual_analysis
            ])
        
        prompt = f"""
        Generate visual recommendations for this marketing content:
        
        Content Type: {content_type}
        Title: {core_content.get('title', '')}
        Main Message: {core_content.get('main_content', '')}
        Key Messages: {core_content.get('key_messages', [])}
        
        Existing Visuals:
        {existing_visual_summary or "No existing visuals"}
        
        Recommend 3-5 visual elements including:
        1. Image style and composition suggestions
        2. Color palette recommendations
        3. Typography suggestions
        4. Layout and design elements
        5. Additional visual assets needed
        
        For each recommendation, provide:
        - type: Type of visual element
        - description: What it should show/convey
        - style: Visual style guidance
        - purpose: How it supports the content
        - placement: Where it should be used
        
        Format as JSON array.
        """
        
        try:
            response = await self.text_llm.ainvoke([HumanMessage(content=prompt)])
            recommendations = json.loads(response.content.strip())
            logger.info(f"Generated {len(recommendations)} visual recommendations")
            return recommendations
        except json.JSONDecodeError:
            logger.warning("Failed to parse visual recommendations")
            return [{"type": "image", "description": "Supporting visual for marketing content"}]
    
    async def _analyze_competitors(self, competitors: List[str], content_type: str) -> Dict[str, Any]:
        """
        Analyze competitors for insights (placeholder for future enhancement)
        
        Args:
            competitors: List of competitor names
            content_type: Type of content being created
            
        Returns:
            Dictionary with competitive insights
        """
        # This is a placeholder - would integrate with web scraping/analysis tools
        return {
            "summary": f"Competitive analysis of {len(competitors)} competitors in {content_type} context",
            "opportunities": ["Unique positioning angle", "Underserved messaging gaps"],
            "differentiation": ["Focus on unique value proposition", "Highlight distinctive benefits"]
        }
    
    def _process_product_info(self, product_data: Dict[str, Any]) -> ProductInfo:
        """Convert dictionary to ProductInfo object"""
        return ProductInfo(
            name=product_data.get("name", "Product"),
            description=product_data.get("description", ""),
            category=product_data.get("category", "General"),
            price=product_data.get("price"),
            key_features=product_data.get("key_features", []),
            benefits=product_data.get("benefits", []),
            target_demographics=product_data.get("target_demographics", []),
            brand_voice=product_data.get("brand_voice", "professional"),
            competitors=product_data.get("competitors", [])
        )
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def _load_local_image(self, path: str) -> Optional[bytes]:
        """Load local image file"""
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading local image {path}: {e}")
            return None
    
    def _create_fallback_audience_insights(self, target_audience: str) -> Dict[str, Any]:
        """Create fallback audience insights"""
        return {
            "demographics": {"description": target_audience},
            "pain_points": ["General customer challenges"],
            "motivations": ["Quality", "Value", "Convenience"],
            "communication_style": "professional and friendly",
            "preferred_channels": ["social_media", "email", "website"],
            "messaging_angles": ["Benefits", "Problem-solving", "Value proposition"],
            "buying_triggers": ["Recommendations", "Reviews", "Special offers"]
        }
    
    def _create_fallback_content(self, product: ProductInfo, content_type: str) -> Dict[str, str]:
        """Create fallback content when generation fails"""
        return {
            "title": f"Discover {product.name}",
            "main_content": f"Experience the benefits of {product.name}. {product.description}",
            "cta": "Learn more today!",
            "key_messages": [product.name, "Quality", "Value"],
            "emotional_hooks": ["Excitement", "Confidence", "Satisfaction"]
        } 
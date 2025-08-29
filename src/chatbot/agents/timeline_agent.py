"""
Timeline Agent

The timeline agent is responsible for understanding One Piece timeline,
chronology, and temporal relationships between events and characters.
"""

from typing import Dict, Any, List, Optional
import re
from datetime import datetime

from .base_agent import BaseAgent, AgentType, AgentInput
from ..config import ChatbotConfig
from ..utils.llm_client import LLMClient


class TimelineAgent(BaseAgent):
    """
    Timeline agent for One Piece chronology and temporal analysis.
    
    This agent:
    - Understands temporal relationships
    - Places events in chronological context
    - Handles timeline-based queries
    - Integrates historical context
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the timeline agent."""
        super().__init__(config, AgentType.TIMELINE)
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Timeline patterns and One Piece era information
        self.timeline_patterns = self._initialize_timeline_patterns()
        self.one_piece_eras = self._initialize_one_piece_eras()
        self.temporal_indicators = self._initialize_temporal_indicators()
    
    def _initialize_timeline_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for timeline detection."""
        return {
            'absolute_time': [
                r'\b(\d{4})\s+(years?|ago)\b',
                r'\b(\d+)\s+(years?|months?|days?|weeks?)\s+(ago|before|earlier)\b',
                r'\b(\d+)\s+(years?|months?|days?|weeks?)\s+(later|after|since)\b',
            ],
            'relative_time': [
                r'\b(before|after|during|while|when|since|until)\b',
                r'\b(earlier|later|previously|subsequently|meanwhile)\b',
                r'\b(first|second|third|last|next|previous|current)\b',
            ],
            'era_indicators': [
                r'\b(era|period|age|time|generation|epoch)\b',
                r'\b(old|new|ancient|modern|current|past|future)\b',
            ],
        }
    
    def _initialize_one_piece_eras(self) -> Dict[str, Dict[str, Any]]:
        """Initialize One Piece timeline eras and periods."""
        return {
            'void_century': {
                'name': 'Void Century',
                'description': 'A mysterious 100-year period erased from history',
                'significance': 'Contains crucial information about the world\'s true history',
                'events': ['Ancient Kingdom destruction', 'World Government formation'],
                'confidence': 0.9,
            },
            'roger_era': {
                'name': 'Age of Roger',
                'description': 'The era when Gol D. Roger was the Pirate King',
                'timeframe': '22+ years ago',
                'events': ['Roger\'s execution', 'Great Pirate Era begins', 'One Piece hidden'],
                'significance': 'Marked the beginning of the current pirate age',
                'confidence': 0.95,
            },
            'great_pirate_era': {
                'name': 'Great Pirate Era',
                'description': 'Current era marked by Roger\'s execution',
                'timeframe': '22+ years and ongoing',
                'events': ['Straw Hat Pirates formation', 'Marineford War', 'Wano Country events'],
                'significance': 'The main timeline of the One Piece story',
                'confidence': 0.95,
            },
            'pre_roger_era': {
                'name': 'Pre-Roger Era',
                'description': 'Era before Roger became Pirate King',
                'timeframe': 'Before Roger\'s rise',
                'events': ['Rocks Pirates era', 'Garp and Roger\'s rivalry'],
                'significance': 'Sets up the world before the main story',
                'confidence': 0.8,
            },
        }
    
    def _initialize_temporal_indicators(self) -> Dict[str, List[str]]:
        """Initialize temporal indicators for timeline analysis."""
        return {
            'past_indicators': [
                'ago', 'before', 'earlier', 'previously', 'was', 'were', 'had',
                'ancient', 'old', 'former', 'past', 'historical'
            ],
            'present_indicators': [
                'now', 'currently', 'present', 'current', 'today', 'is', 'are',
                'ongoing', 'happening', 'active'
            ],
            'future_indicators': [
                'will', 'going to', 'future', 'upcoming', 'planned', 'expected',
                'soon', 'later', 'next', 'forthcoming'
            ],
            'sequence_indicators': [
                'first', 'second', 'third', 'then', 'next', 'after', 'before',
                'during', 'while', 'meanwhile', 'simultaneously'
            ],
        }
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the timeline agent logic.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Dictionary containing timeline analysis and chronological information
        """
        self.logger.info("Executing timeline analysis and chronology")
        
        # Extract search results from context
        search_results = self._extract_search_results(input_data)
        
        # Analyze timeline aspects of the query
        timeline_analysis = self._analyze_timeline_query(input_data.query)
        
        # Extract temporal information from search results using LLM
        llm_timeline = self._extract_timeline_with_llm(search_results, input_data)
        temporal_info = llm_timeline.get('temporal_information', {})
        timeline_events = llm_timeline.get('timeline_events', [])
        chronological_context = llm_timeline.get('chronological_context', '')
        self.logger.info("Timeline analysis completed using LLM")
        
        # Compile timeline output
        timeline_output = {
            'query_timeline_focus': timeline_analysis,
            'temporal_information': temporal_info,
            'timeline_events': timeline_events,
            'chronological_context': chronological_context,
            'one_piece_eras': self._get_relevant_eras(input_data.query),
            'confidence_score': self._calculate_timeline_confidence(timeline_events, temporal_info),
            'metadata': {
                'search_results_analyzed': len(search_results),
                'temporal_indicators_found': len(temporal_info.get('indicators', [])),
                'timeline_events_identified': len(timeline_events),
                'era_relevance': len(self._get_relevant_eras(input_data.query)),
                'llm_used': True,
            }
        }
        
        self.logger.info("Timeline analysis completed successfully")
        
        return timeline_output
    
    def _extract_timeline_with_llm(self, search_results: List[Dict[str, Any]], 
                                  input_data: AgentInput) -> Dict[str, Any]:
        """Extract timeline information using LLM for enhanced analysis."""
        try:
            if not self.llm_client or not self.llm_client.is_available():
                return {}
            
            # Use LLM for timeline analysis
            llm_result = self.llm_client.generate_timeline_analysis(
                query=input_data.query,
                search_results=search_results
            )
            
            # Convert LLM result to expected format
            return {
                'temporal_information': {
                    'indicators': ['LLM-extracted temporal indicators'],
                    'time_periods': ['LLM-identified time periods'],
                    'sequence_markers': ['LLM-identified sequence markers'],
                },
                'timeline_events': [llm_result.get('timeline_events', 'LLM-generated timeline')],
                'chronological_context': llm_result.get('historical_context', 'LLM-generated context'),
            }
            
        except Exception as e:
            self.logger.warning(f"LLM timeline extraction failed: {e}")
            return {}
    
    def _extract_search_results(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """Extract search results from the input context."""
        if not input_data.context:
            return []
        
        # Look for search agent output
        search_output = input_data.context.get('search_agent', {})
        if isinstance(search_output, dict) and 'results' in search_output:
            return search_output['results']
        
        # Look for direct search results
        if 'search_results' in input_data.context:
            return search_output['search_results']
        
        return []
    
    def _analyze_timeline_query(self, query: str) -> Dict[str, Any]:
        """Analyze the timeline aspects of the user query."""
        query_lower = query.lower()
        
        timeline_focus = {
            'has_temporal_focus': False,
            'temporal_type': 'none',
            'specific_time_period': None,
            'relative_timing': None,
            'era_interest': None,
        }
        
        # Check for temporal focus
        if any(word in query_lower for word in ['when', 'time', 'timeline', 'chronology', 'history']):
            timeline_focus['has_temporal_focus'] = True
        
        # Determine temporal type
        if any(word in query_lower for word in ['when', 'time']):
            timeline_focus['temporal_type'] = 'specific_time'
        elif any(word in query_lower for word in ['timeline', 'chronology', 'order']):
            timeline_focus['temporal_type'] = 'sequence'
        elif any(word in query_lower for word in ['history', 'era', 'period']):
            timeline_focus['temporal_type'] = 'historical_context'
        
        # Check for specific time periods
        for era_name, era_info in self.one_piece_eras.items():
            if era_name.replace('_', ' ') in query_lower or era_info['name'].lower() in query_lower:
                timeline_focus['era_interest'] = era_name
                break
        
        # Check for relative timing
        if any(word in query_lower for word in ['before', 'after', 'during', 'while']):
            timeline_focus['relative_timing'] = 'relative'
        elif any(word in query_lower for word in ['ago', 'years', 'months', 'days']):
            timeline_focus['relative_timing'] = 'absolute'
        
        return timeline_focus
    
    def _extract_temporal_information(self, search_results: List[Dict[str, Any]], 
                                    query: str) -> Dict[str, Any]:
        """Extract temporal information from search results."""
        temporal_info = {
            'indicators': [],
            'time_periods': [],
            'relative_events': [],
            'absolute_dates': [],
            'era_mentions': [],
        }
        
        for result in search_results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Extract temporal indicators
            indicators = self._extract_temporal_indicators(content)
            if indicators:
                temporal_info['indicators'].extend(indicators)
            
            # Extract time periods
            time_periods = self._extract_time_periods(content)
            if time_periods:
                temporal_info['time_periods'].extend(time_periods)
            
            # Extract relative events
            relative_events = self._extract_relative_events(content)
            if relative_events:
                temporal_info['relative_events'].extend(relative_events)
            
            # Extract absolute dates
            absolute_dates = self._extract_absolute_dates(content)
            if absolute_dates:
                temporal_info['absolute_dates'].extend(absolute_dates)
            
            # Extract era mentions
            era_mentions = self._extract_era_mentions(content)
            if era_mentions:
                temporal_info['era_mentions'].extend(era_mentions)
        
        return temporal_info
    
    def _extract_temporal_indicators(self, content: str) -> List[Dict[str, Any]]:
        """Extract temporal indicators from content."""
        indicators = []
        
        for indicator_type, patterns in self.temporal_indicators.items():
            for pattern in patterns:
                if pattern in content.lower():
                    indicators.append({
                        'type': indicator_type,
                        'indicator': pattern,
                        'context': content[:100] + "..." if len(content) > 100 else content,
                    })
        
        return indicators
    
    def _extract_time_periods(self, content: str) -> List[Dict[str, Any]]:
        """Extract time periods from content."""
        time_periods = []
        
        # Look for time period patterns
        period_patterns = [
            (r'(\d+)\s+(years?|months?|days?|weeks?)\s+(ago|before|earlier)', 'past_period'),
            (r'(\d+)\s+(years?|months?|days?|weeks?)\s+(later|after|since)', 'future_period'),
            (r'(century|decade|era|period|age)', 'general_period'),
        ]
        
        for pattern, period_type in period_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                time_periods.append({
                    'type': period_type,
                    'value': match.group(1) if match.group(1).isdigit() else match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 and match.group(2) else None,
                    'direction': match.group(3) if len(match.groups()) > 2 else None,
                    'context': content[:100] + "..." if len(content) > 100 else content,
                })
        
        return time_periods
    
    def _extract_relative_events(self, content: str) -> List[Dict[str, Any]]:
        """Extract relative event timing from content."""
        relative_events = []
        
        # Look for relative timing patterns
        relative_patterns = [
            (r'(before|after|during|while)\s+([^,\.]+)', 'relative_timing'),
            (r'([^,\.]+)\s+(before|after|during|while)\s+([^,\.]+)', 'event_sequence'),
        ]
        
        for pattern, event_type in relative_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if event_type == 'relative_timing':
                    relative_events.append({
                        'type': event_type,
                        'timing': match.group(1),
                        'event': match.group(2).strip(),
                        'context': content[:100] + "..." if len(content) > 100 else content,
                    })
                else:
                    relative_events.append({
                        'type': event_type,
                        'event1': match.group(1).strip(),
                        'timing': match.group(2),
                        'event2': match.group(3).strip(),
                        'context': content[:100] + "..." if len(content) > 100 else content,
                    })
        
        return relative_events
    
    def _extract_absolute_dates(self, content: str) -> List[Dict[str, Any]]:
        """Extract absolute dates from content."""
        absolute_dates = []
        
        # Look for year patterns
        year_patterns = [
            (r'(\d{4})', 'year'),
            (r'(\d{1,2})\s+(years?)\s+(ago)', 'years_ago'),
        ]
        
        for pattern, date_type in year_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if date_type == 'year':
                    year = int(match.group(1))
                    if 1500 <= year <= 2100:  # Reasonable year range
                        absolute_dates.append({
                            'type': date_type,
                            'value': year,
                            'context': content[:100] + "..." if len(content) > 100 else content,
                        })
                elif date_type == 'years_ago':
                    years = int(match.group(1))
                    absolute_dates.append({
                        'type': date_type,
                        'value': years,
                        'context': content[:100] + "..." if len(content) > 100 else content,
                    })
        
        return absolute_dates
    
    def _extract_era_mentions(self, content: str) -> List[Dict[str, Any]]:
        """Extract mentions of One Piece eras from content."""
        era_mentions = []
        
        for era_key, era_info in self.one_piece_eras.items():
            era_name = era_info['name']
            if era_name.lower() in content.lower():
                era_mentions.append({
                    'era_key': era_key,
                    'era_name': era_name,
                    'description': era_info['description'],
                    'context': content[:100] + "..." if len(content) > 100 else content,
                })
        
        return era_mentions
    
    def _identify_timeline_events(self, search_results: List[Dict[str, Any]], 
                                temporal_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify timeline events and their chronological order."""
        timeline_events = []
        
        # Process absolute dates
        for date_info in temporal_info.get('absolute_dates', []):
            if date_info['type'] == 'year':
                timeline_events.append({
                    'type': 'absolute_date',
                    'date': date_info['value'],
                    'description': f"Event in {date_info['value']}",
                    'confidence': 0.9,
                    'source': 'temporal_analysis',
                })
            elif date_info['type'] == 'years_ago':
                # Calculate approximate year (assuming current year is story present)
                current_year = 2024  # This would be dynamic based on story timeline
                event_year = current_year - date_info['value']
                timeline_events.append({
                    'type': 'calculated_date',
                    'date': event_year,
                    'description': f"Event {date_info['value']} years ago",
                    'confidence': 0.7,
                    'source': 'temporal_analysis',
                })
        
        # Process relative events
        for event_info in temporal_info.get('relative_events', []):
            if event_info['type'] == 'event_sequence':
                timeline_events.append({
                    'type': 'relative_sequence',
                    'event1': event_info['event1'],
                    'timing': event_info['timing'],
                    'event2': event_info['event2'],
                    'description': f"{event_info['event1']} {event_info['timing']} {event_info['event2']}",
                    'confidence': 0.8,
                    'source': 'temporal_analysis',
                })
        
        # Add One Piece era events
        for era_mention in temporal_info.get('era_mentions', []):
            era_key = era_mention['era_key']
            era_info = self.one_piece_eras[era_key]
            
            if 'events' in era_info:
                for event in era_info['events']:
                    timeline_events.append({
                        'type': 'era_event',
                        'era': era_key,
                        'event': event,
                        'description': f"{event} during {era_info['name']}",
                        'confidence': era_info.get('confidence', 0.8),
                        'source': 'one_piece_era',
                    })
        
        # Sort events by confidence and type
        timeline_events.sort(key=lambda x: (x['confidence'], x['type'] != 'absolute_date'), reverse=True)
        
        return timeline_events
    
    def _generate_chronological_context(self, timeline_events: List[Dict[str, Any]], 
                                     query: str) -> Dict[str, Any]:
        """Generate chronological context for the timeline events."""
        chronological_context = {
            'timeline_summary': '',
            'key_periods': [],
            'event_sequence': [],
            'temporal_gaps': [],
            'chronological_insights': [],
        }
        
        if not timeline_events:
            chronological_context['timeline_summary'] = "No timeline information found in the search results."
            return chronological_context
        
        # Generate timeline summary
        summary_parts = []
        if timeline_events:
            summary_parts.append(f"Found {len(timeline_events)} timeline events")
            
            # Group by type
            event_types = {}
            for event in timeline_events:
                event_type = event['type']
                if event_type not in event_types:
                    event_types[event_type] = []
                event_types[event_type].append(event)
            
            for event_type, events in event_types.items():
                summary_parts.append(f"{len(events)} {event_type.replace('_', ' ')} events")
        
        chronological_context['timeline_summary'] = ". ".join(summary_parts) + "."
        
        # Identify key periods
        key_periods = self._identify_key_periods(timeline_events)
        chronological_context['key_periods'] = key_periods
        
        # Generate event sequence
        event_sequence = self._generate_event_sequence(timeline_events)
        chronological_context['event_sequence'] = event_sequence
        
        # Identify temporal gaps
        temporal_gaps = self._identify_temporal_gaps(timeline_events)
        chronological_context['temporal_gaps'] = temporal_gaps
        
        # Generate chronological insights
        chronological_insights = self._generate_chronological_insights(timeline_events, query)
        chronological_context['chronological_insights'] = chronological_insights
        
        return chronological_context
    
    def _identify_key_periods(self, timeline_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key time periods from timeline events."""
        key_periods = []
        
        # Group events by approximate time periods
        time_groups = {}
        for event in timeline_events:
            if 'date' in event:
                # Group by decades
                decade = (event['date'] // 10) * 10
                if decade not in time_groups:
                    time_groups[decade] = []
                time_groups[decade].append(event)
        
        # Identify periods with multiple events
        for decade, events in time_groups.items():
            if len(events) > 1:
                key_periods.append({
                    'period': f"{decade}s",
                    'event_count': len(events),
                    'events': [event['description'] for event in events[:3]],  # Top 3 events
                    'significance': 'high' if len(events) > 2 else 'medium',
                })
        
        return key_periods
    
    def _generate_event_sequence(self, timeline_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a sequence of events in chronological order."""
        # Filter events with dates
        dated_events = [event for event in timeline_events if 'date' in event]
        
        # Sort by date
        dated_events.sort(key=lambda x: x['date'])
        
        # Generate sequence
        sequence = []
        for i, event in enumerate(dated_events):
            sequence.append({
                'order': i + 1,
                'date': event['date'],
                'description': event['description'],
                'type': event['type'],
                'confidence': event['confidence'],
            })
        
        return sequence
    
    def _identify_temporal_gaps(self, timeline_events: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps in the timeline information."""
        gaps = []
        
        # Check for gaps between dated events
        dated_events = [event for event in timeline_events if 'date' in event]
        if len(dated_events) > 1:
            dated_events.sort(key=lambda x: x['date'])
            
            for i in range(len(dated_events) - 1):
                gap = dated_events[i + 1]['date'] - dated_events[i]['date']
                if gap > 50:  # Gap of more than 50 years
                    gaps.append(f"Large gap between {dated_events[i]['date']} and {dated_events[i + 1]['date']} ({gap} years)")
        
        # Check for missing era coverage
        covered_eras = set()
        for event in timeline_events:
            if event['type'] == 'era_event':
                covered_eras.add(event['era'])
        
        missing_eras = set(self.one_piece_eras.keys()) - covered_eras
        if missing_eras:
            gaps.append(f"Missing information for eras: {', '.join(missing_eras)}")
        
        return gaps
    
    def _generate_chronological_insights(self, timeline_events: List[Dict[str, Any]], 
                                       query: str) -> List[Dict[str, Any]]:
        """Generate insights about the chronological information."""
        insights = []
        
        # Insight about event distribution
        if timeline_events:
            dated_events = [event for event in timeline_events if 'date' in event]
            if dated_events:
                date_range = max(event['date'] for event in dated_events) - min(event['date'] for event in dated_events)
                insights.append({
                    'type': 'temporal_span',
                    'insight': f"Events span {date_range} years",
                    'confidence': 0.9,
                })
        
        # Insight about era coverage
        era_events = [event for event in timeline_events if event['type'] == 'era_event']
        if era_events:
            unique_eras = len(set(event['era'] for event in era_events))
            insights.append({
                'type': 'era_coverage',
                'insight': f"Information covers {unique_eras} different eras",
                'confidence': 0.8,
            })
        
        # Query-specific insights
        query_lower = query.lower()
        if 'timeline' in query_lower or 'chronology' in query_lower:
            if timeline_events:
                insights.append({
                    'type': 'timeline_completeness',
                    'insight': f"Timeline analysis identified {len(timeline_events)} key events",
                    'confidence': 0.8,
                })
        
        return insights
    
    def _get_relevant_eras(self, query: str) -> List[Dict[str, Any]]:
        """Get eras relevant to the user's query."""
        relevant_eras = []
        query_lower = query.lower()
        
        for era_key, era_info in self.one_piece_eras.items():
            # Check if era name or key terms appear in query
            if (era_info['name'].lower() in query_lower or 
                era_key.replace('_', ' ') in query_lower):
                relevant_eras.append({
                    'era_key': era_key,
                    'name': era_info['name'],
                    'description': era_info['description'],
                    'timeframe': era_info.get('timeframe', 'Unknown'),
                    'events': era_info.get('events', []),
                    'significance': era_info.get('significance', ''),
                    'confidence': era_info.get('confidence', 0.8),
                })
        
        return relevant_eras
    
    def _calculate_timeline_confidence(self, timeline_events: List[Dict[str, Any]], 
                                     temporal_info: Dict[str, Any]) -> float:
        """Calculate confidence score for the timeline analysis."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on timeline events
        if timeline_events:
            confidence += min(0.3, len(timeline_events) * 0.05)
        
        # Adjust based on temporal information
        if temporal_info.get('indicators'):
            confidence += min(0.1, len(temporal_info['indicators']) * 0.02)
        
        if temporal_info.get('absolute_dates'):
            confidence += min(0.1, len(temporal_info['absolute_dates']) * 0.02)
        
        # Adjust based on era coverage
        if temporal_info.get('era_mentions'):
            confidence += min(0.1, len(temporal_info['era_mentions']) * 0.02)
        
        return min(confidence, 1.0)
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the timeline agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Timeline agent requires a query
        if not input_data.query:
            self.logger.warning("Timeline agent requires a query")
            return False
        
        return True

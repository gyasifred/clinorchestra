#!/usr/bin/env python3
"""
Extras Manager - Keyword-based matching for supplementary hints
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
4. Extras are now supplementary knowledge to help LLM break down tasks
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from core.logging_config import get_logger

logger = get_logger(__name__)


class ExtrasManager:
    """
    Manage extras (supplementary hints/tips/patterns) to help LLM understand tasks
    
    Extras are NOT for matching against input text - they're supplementary knowledge
    that helps the LLM better understand how to approach the extraction task.
    
    Examples:
    - Diagnostic criteria explanations
    - Assessment methodology tips
    - Common patterns in specific domains
    - Task breakdown guidance
    - Field interpretation hints
    """
    
    def __init__(self, storage_path: str = "./extras"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.extras: List[Dict[str, Any]] = []
        
        self._load_all_extras()
        
        logger.info(f"ExtrasManager initialized with {len(self.extras)} extras")
    
    def add_extra(self, extra_type: str, content: str, metadata: Optional[Dict] = None, name: Optional[str] = None) -> bool:
        """Add extra (hint/tip/pattern) to storage with optional name"""
        try:
            # Auto-generate name from content if not provided
            if not name or not name.strip():
                name = self._generate_name_from_content(content, extra_type)

            extra = {
                'id': f"extra_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                'name': name.strip(),
                'type': extra_type,
                'content': content,
                'metadata': metadata or {},
                'enabled': True,  # New: extras are enabled by default
                'created_at': datetime.now().isoformat()
            }

            self.extras.append(extra)
            self._save_extra(extra)

            logger.info(f"Added extra: {name} ({extra['id']})")
            return True

        except Exception as e:
            logger.error(f"Failed to add extra: {e}")
            return False

    def _generate_name_from_content(self, content: str, extra_type: str) -> str:
        """Generate a meaningful name from content"""
        # Take first 40 chars and clean up
        name = content[:40].strip()

        # Remove newlines and extra spaces
        name = ' '.join(name.split())

        # Add ellipsis if truncated
        if len(content) > 40:
            name += "..."

        # Prefix with type for clarity
        type_prefix = extra_type.capitalize()
        name = f"{type_prefix}: {name}"

        return name
    
    def match_extras_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        ENHANCED: Match extras based on keywords from the extraction task
        
        This is the correct usage: keywords come from schema fields, task labels,
        and task description - NOT from input text.
        
        Args:
            keywords: List of keywords from extraction task (schema fields, labels, etc.)
            
        Returns:
            List of matched extras (hints/tips/patterns) sorted by relevance
        """
        if not keywords:
            logger.warning("No keywords provided for extras matching")
            return []
        
        if not self.extras:
            logger.info("No extras available in storage")
            return []
        
        logger.info(f"Matching extras against keywords: {keywords}")

        matched = []
        keywords_lower = [k.lower() for k in keywords if k]

        # Filter to only use enabled extras for matching
        enabled_extras = [e for e in self.extras if e.get('enabled', True)]
        logger.info(f"Using {len(enabled_extras)} enabled extras out of {len(self.extras)} total")

        for extra in enabled_extras:
            score = self._calculate_keyword_relevance_score(extra, keywords_lower)
            
            if score > 0.2:  # Threshold for relevance
                matched.append({
                    **extra,
                    'relevance_score': score,
                    'matched_keywords': self._get_matched_keywords(extra, keywords_lower)
                })
        
        # Sort by relevance score
        matched.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Matched {len(matched)} extras with keyword relevance > 0.2")

        # Return top 10 most relevant (increased from 5 for better context)
        return matched[:10]
    
    def match_extras(self, text: str, current_output: Optional[Dict[str, Any]] = None, 
                     keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        LEGACY: Match extras using both keywords and text
        
        This method maintains backward compatibility but now prioritizes keyword matching.
        If keywords are provided, use keyword-based matching (correct usage).
        Otherwise, fall back to text-based matching (legacy behavior).
        
        Args:
            text: Input text (legacy parameter)
            current_output: Current extraction output (legacy parameter)
            keywords: Keywords from extraction task (preferred parameter)
            
        Returns:
            List of matched extras sorted by relevance
        """
        # If keywords provided, use keyword-based matching (correct approach)
        if keywords:
            return self.match_extras_by_keywords(keywords)
        
        # Legacy behavior: text-based matching
        logger.info("Using legacy text-based extras matching (keywords preferred)")
        
        if not text:
            return []

        # Filter to only use enabled extras for matching
        enabled_extras = [e for e in self.extras if e.get('enabled', True)]

        matched = []
        text_lower = text.lower()

        output_str = ""
        if current_output:
            output_str = json.dumps(current_output).lower()

        for extra in enabled_extras:
            score = self._calculate_text_relevance_score(
                extra,
                text_lower,
                output_str
            )
            
            if score > 0.5:
                matched.append({
                    **extra,
                    'relevance_score': score
                })
        
        matched.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return top 7 most relevant (increased from 5 for better context)
        return matched[:7]
    
    def _calculate_keyword_relevance_score(self, extra: Dict[str, Any], 
                                          keywords_lower: List[str]) -> float:
        """
        ENHANCED: Calculate relevance score based on keyword matching
        
        This scoring prioritizes:
        1. Exact keyword matches in content
        2. Partial keyword matches
        3. Keyword matches in metadata/type
        """
        content = extra.get('content', '').lower()
        extra_type = extra.get('type', '').lower()
        metadata = extra.get('metadata', {})
        
        if not content:
            return 0.0
        
        score = 0.0
        matched_count = 0
        
        # Extract words from content for matching
        content_words = set(content.split())
        
        for keyword in keywords_lower:
            if not keyword:
                continue
            
            # Exact keyword match in content (high score)
            if keyword in content:
                score += 1.0
                matched_count += 1
            # Partial match: keyword is part of a word
            elif any(keyword in word for word in content_words):
                score += 0.5
                matched_count += 1
            # Keyword in type field
            elif keyword in extra_type:
                score += 0.3
                matched_count += 1
            # Keyword in metadata
            elif any(keyword in str(v).lower() for v in metadata.values()):
                score += 0.2
                matched_count += 1
        
        # Normalize by number of keywords
        if keywords_lower:
            score = score / len(keywords_lower)
        
        # Bonus for matching multiple keywords
        if matched_count > 1:
            score *= 1.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_matched_keywords(self, extra: Dict[str, Any], 
                             keywords_lower: List[str]) -> List[str]:
        """Get list of keywords that matched this extra"""
        content = extra.get('content', '').lower()
        extra_type = extra.get('type', '').lower()
        
        matched = []
        
        for keyword in keywords_lower:
            if keyword and (keyword in content or keyword in extra_type):
                matched.append(keyword)
        
        return matched
    
    def _calculate_text_relevance_score(self, extra: Dict[str, Any], 
                                       text_lower: str, output_str: str) -> float:
        """
        LEGACY: Calculate relevance score for text-based matching
        
        This is the old method - kept for backward compatibility
        """
        content = extra.get('content', '').lower()
        
        if not content:
            return 0.0
        
        keywords = [w for w in content.split() if len(w) > 3]
        
        if not keywords:
            return 0.0
        
        text_matches = sum(1 for kw in keywords if kw in text_lower)
        
        output_matches = sum(1 for kw in keywords if kw in output_str) if output_str else 0
        
        score = (text_matches * 0.7 + output_matches * 0.3) / len(keywords)
        
        return min(score, 1.0)
    
    def list_extras(self) -> List[Dict[str, Any]]:
        """List all extras"""
        return self.extras
    
    def get_extra(self, extra_id: str) -> Optional[Dict[str, Any]]:
        """Get specific extra by ID"""
        for extra in self.extras:
            if extra['id'] == extra_id:
                return extra
        return None
    
    def update_extra(self, extra_id: str, extra_type: str, content: str, metadata: Optional[Dict] = None, name: Optional[str] = None) -> bool:
        """Update an existing extra"""
        try:
            # Find the extra
            extra = self.get_extra(extra_id)
            if not extra:
                logger.error(f"Extra {extra_id} not found")
                return False

            # Auto-generate name if not provided
            if not name or not name.strip():
                name = self._generate_name_from_content(content, extra_type)

            # Update the extra
            extra['name'] = name.strip()
            extra['type'] = extra_type
            extra['content'] = content
            extra['metadata'] = metadata or {}

            # Save updated extra
            self._save_extra(extra)

            logger.info(f"Updated extra: {name} ({extra_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to update extra: {e}")
            return False

    def remove_extra(self, extra_id: str) -> bool:
        """Remove extra by ID"""
        try:
            self.extras = [e for e in self.extras if e['id'] != extra_id]

            extra_file = self.storage_path / f"{extra_id}.json"
            if extra_file.exists():
                extra_file.unlink()

            logger.info(f"Removed extra: {extra_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove extra: {e}")
            return False

    def enable_extra(self, extra_id: str, enabled: bool = True) -> bool:
        """Enable or disable an extra"""
        try:
            extra = self.get_extra(extra_id)
            if not extra:
                logger.error(f"Extra {extra_id} not found")
                return False

            extra['enabled'] = enabled
            self._save_extra(extra)

            status = "enabled" if enabled else "disabled"
            logger.info(f"Extra {extra_id} {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable/disable extra: {e}")
            return False
    
    def _save_extra(self, extra: Dict[str, Any]):
        """Save extra to file"""
        extra_file = self.storage_path / f"{extra['id']}.json"
        with open(extra_file, 'w') as f:
            json.dump(extra, f, indent=2)
    
    def _load_all_extras(self):
        """Load all extras from storage with backward compatibility for name and id fields"""
        for extra_file in self.storage_path.glob("*.json"):
            try:
                with open(extra_file, 'r') as f:
                    extra = json.load(f)

                    # Backward compatibility: add id if missing
                    if 'id' not in extra:
                        # Use filename as ID if it follows the extra_* pattern
                        file_id = extra_file.stem
                        if file_id.startswith('extra_'):
                            extra['id'] = file_id
                        else:
                            # Generate new ID
                            extra['id'] = f"extra_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                        logger.info(f"Added missing 'id' field to {extra_file.name}: {extra['id']}")
                        # Save with new id field
                        self._save_extra(extra)

                    # Backward compatibility: add name if missing
                    if 'name' not in extra:
                        extra['name'] = self._generate_name_from_content(
                            extra.get('content', ''),
                            extra.get('type', 'hint')
                        )
                        # Save with new name field
                        self._save_extra(extra)

                    # Backward compatibility: add enabled field if missing (default to True)
                    if 'enabled' not in extra:
                        extra['enabled'] = True
                        self._save_extra(extra)

                    self.extras.append(extra)
            except Exception as e:
                logger.error(f"Failed to load {extra_file}: {e}")
    
    def export_extras(self) -> str:
        """Export all extras as JSON"""
        return json.dumps(self.extras, indent=2)
    
    def import_extras(self, json_str: str) -> bool:
        """Import extras from JSON"""
        try:
            extras = json.loads(json_str)
            for extra in extras:
                self.extras.append(extra)
                self._save_extra(extra)
            return True
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
    
    def search_extras(self, query: str) -> List[Dict[str, Any]]:
        """
        Search extras by query string
        
        Args:
            query: Search query
            
        Returns:
            List of matching extras
        """
        if not query:
            return self.extras
        
        query_lower = query.lower()
        results = []
        
        for extra in self.extras:
            content = extra.get('content', '').lower()
            extra_type = extra.get('type', '').lower()
            
            if query_lower in content or query_lower in extra_type:
                results.append(extra)
        
        return results
    
    def get_extras_by_type(self, extra_type: str) -> List[Dict[str, Any]]:
        """
        Get all extras of a specific type
        
        Args:
            extra_type: Type of extras to retrieve
            
        Returns:
            List of extras of the specified type
        """
        return [e for e in self.extras if e.get('type', '').lower() == extra_type.lower()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extras statistics"""
        types = {}
        for extra in self.extras:
            extra_type = extra.get('type', 'unknown')
            types[extra_type] = types.get(extra_type, 0) + 1
        
        return {
            'total_extras': len(self.extras),
            'types': types,
            'storage_path': str(self.storage_path)
        }
#!/usr/bin/env python3
"""
Extras Manager - Keyword-based matching for supplementary hints
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
4. Extras are now supplementary knowledge to help LLM break down tasks
"""

import json
import yaml
import re
from typing import Dict, Any, List, Optional, Tuple
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

        # PERFORMANCE: Result caching for repeated keyword queries
        self.result_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_bypass = False  # Can be set to True to force recompute

        self._load_all_extras()

        logger.info(f"ExtrasManager initialized with {len(self.extras)} extras")
    
    def add_extra(self, extra_type: str, content: str, metadata: Optional[Dict] = None, name: Optional[str] = None) -> bool:
        """Add extra (hint/tip/pattern) to storage with optional name"""
        try:
            # Auto-generate name from content if not provided
            if not name or not name.strip():
                name = self._generate_name_from_content(content, extra_type)

            # CRITICAL FIX: Check for duplicate by name to prevent duplicates on instance restart
            existing_extra = self._get_extra_by_name(name.strip())
            if existing_extra:
                logger.info(f"Extra with name '{name}' already exists (ID: {existing_extra['id']}). Skipping duplicate.")
                return True  # Return success since the extra already exists

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

            # Clear cache since extras have changed
            self.result_cache.clear()

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

        PERFORMANCE: Results are cached based on keywords to avoid redundant computation.

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

        # PERFORMANCE: Create cache key from sorted keywords
        keywords_lower = [k.lower() for k in keywords if k]
        cache_key = "|".join(sorted(keywords_lower))

        # Check cache first (unless bypass is enabled)
        if not self.cache_bypass and cache_key in self.result_cache:
            self.cache_hits += 1
            cached_result = self.result_cache[cache_key]
            logger.debug(f"Cache HIT for extras keywords {keywords} - returning {len(cached_result)} cached results")
            return cached_result

        self.cache_misses += 1
        logger.info(f"Matching extras against keywords: {keywords}")

        matched = []

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
        result = matched[:10]

        # Cache the result
        self.result_cache[cache_key] = result
        logger.debug(f"Cached extras result for keywords {keywords}")

        return result

    def match_extras_with_variations(self,
                                     core_keywords: List[str],
                                     variations_map: Optional[Dict[str, List[str]]] = None,
                                     threshold: float = 0.2,
                                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Match extras using core keywords + term variations for improved recall and leniency.

        This method implements the search strategy by expanding each core concept with
        its variations (synonyms, abbreviations, related terms) to increase the chance
        of finding relevant extras.

        Args:
            core_keywords: List of core concept keywords
            variations_map: Dict mapping core keywords to their variations.
                           If None, uses core_keywords directly.
            threshold: Minimum relevance score (default: 0.2)
            top_k: Maximum results to return (default: 10)

        Returns:
            List of matched extras sorted by relevance score

        Example:
            core_keywords = ["[condition_A]", "[domain_B]"]
            variations_map = {
                "[condition_A]": ["[synonym1]", "[abbrev1]", "[abbrev2]", "[related1]"],
                "[domain_B]": ["[related_B1]", "[related_B2]", "[qualified_B]"]
            }
            results = manager.match_extras_with_variations(core_keywords, variations_map)
        """
        if not core_keywords:
            logger.warning("No core keywords provided for extras matching")
            return []

        # Build expanded keyword list
        all_keywords = set(core_keywords)

        if variations_map:
            for core in core_keywords:
                if core in variations_map:
                    variations = variations_map.get(core, [])
                    all_keywords.update(variations)
                    logger.debug(f"Expanded '{core}' with {len(variations)} variations")

        expanded_keywords = list(all_keywords)
        logger.info(f"Extras matching: {len(core_keywords)} core → {len(expanded_keywords)} total keywords")

        # Use existing match_extras_by_keywords with expanded set
        return self.match_extras_by_keywords(expanded_keywords)

    def clear_cache(self):
        """Clear the result cache"""
        self.result_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("ExtrasManager result cache cleared")

    def set_cache_bypass(self, bypass: bool):
        """Set cache bypass mode"""
        self.cache_bypass = bypass
        logger.info(f"ExtrasManager cache bypass set to: {bypass}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.result_cache),
            'bypass_enabled': self.cache_bypass
        }
    
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

    def _get_extra_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific extra by name (for duplicate checking)"""
        for extra in self.extras:
            if extra.get('name', '').strip() == name.strip():
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

            # Clear cache since extras have changed
            self.result_cache.clear()

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

            # Clear cache since extras have changed
            self.result_cache.clear()

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

            # Clear cache since extras have changed
            self.result_cache.clear()

            status = "enabled" if enabled else "disabled"
            logger.info(f"Extra {extra_id} {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable/disable extra: {e}")
            return False
    
    def _save_extra(self, extra: Dict[str, Any]):
        """Save extra to file using extras name (not random UID)"""
        # Use extras name for filename, sanitize it for filesystem safety
        name = extra.get('name', extra.get('id', 'unnamed_extra'))
        # Sanitize filename: remove/replace invalid characters
        safe_name = re.sub(r'[^\w\s-]', '_', name).strip().replace(' ', '_')
        extra_file = self.storage_path / f"{safe_name}.json"

        with open(extra_file, 'w') as f:
            json.dump(extra, f, indent=2)
    
    def _load_all_extras(self):
        """Load all extras from storage with backward compatibility for name and id fields"""
        loaded_names = set()  # Track loaded names to prevent duplicates

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

                    # CRITICAL FIX: Check for duplicate by name before adding
                    extra_name = extra.get('name', '').strip()
                    if extra_name in loaded_names:
                        logger.warning(f"Skipping duplicate extra '{extra_name}' from {extra_file.name} (already loaded)")
                        continue

                    loaded_names.add(extra_name)
                    self.extras.append(extra)
            except Exception as e:
                logger.error(f"Failed to load {extra_file}: {e}")
    
    def export_extras(self) -> str:
        """Export all extras as JSON"""
        return json.dumps(self.extras, indent=2)
    
    def import_extras(self, yaml_str: str) -> Tuple[bool, int, str]:
        """Import extras from YAML or JSON string, stores internally as JSON"""
        try:
            # Try YAML first (supports both YAML and JSON since JSON is valid YAML)
            extras = yaml.safe_load(yaml_str)

            if not isinstance(extras, list):
                return False, 0, "Invalid format: expected list of extras"

            count = 0
            errors = []

            for extra in extras:
                try:
                    # Ensure required fields
                    if 'content' not in extra:
                        errors.append(f"Extra missing 'content' field")
                        continue

                    # Ensure enabled field exists (backward compatibility)
                    if 'enabled' not in extra:
                        extra['enabled'] = True

                    # Ensure id field (use name if available, otherwise generate)
                    if 'id' not in extra:
                        if 'name' in extra:
                            extra['id'] = extra['name']
                        else:
                            extra['id'] = f"extra_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

                    self.extras.append(extra)
                    self._save_extra(extra)
                    count += 1
                except Exception as e:
                    errors.append(f"Failed to import extra: {str(e)}")
                    continue

            if errors:
                error_msg = "; ".join(errors[:3])  # Show first 3 errors
                if len(errors) > 3:
                    error_msg += f" ... and {len(errors) - 3} more"
                return count > 0, count, f"Imported {count} extras with {len(errors)} errors: {error_msg}"

            return True, count, f"Successfully imported {count} extras"
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False, 0, f"Import failed: {str(e)}"
    
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
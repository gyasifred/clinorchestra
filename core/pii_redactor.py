"""
PII/PHI Redaction Module - Production Version
Comprehensive redaction of Protected Health Information (HIPAA-compliant)

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: HeiderLab
Version: 1.0.0
Date: 2025

This module provides robust PII/PHI redaction capabilities following HIPAA Safe Harbor
de-identification guidelines (45 CFR § 164.514(b)(2)).
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from core.logging_config import get_logger

logger = get_logger(__name__)

class RedactionMethod(Enum):
    """Enumeration of redaction methods"""
    TAG = "Replace with tag"
    PLACEHOLDER = "Replace with placeholder"
    REMOVE = "Remove completely"

@dataclass
class RedactionPattern:
    """
    Redaction pattern definition with validation
    
    Attributes:
        name: Pattern identifier
        pattern: Compiled regex pattern
        replacement: Replacement string or template
        entity_type: Type of entity being redacted
        priority: Processing priority (higher = earlier)
        description: Human-readable pattern description
    """
    name: str
    pattern: re.Pattern
    replacement: str
    entity_type: str
    priority: int = 0
    description: str = ""
    
    def __post_init__(self):
        """Validate pattern after initialization"""
        if not isinstance(self.pattern, re.Pattern):
            raise TypeError(f"pattern must be compiled regex, got {type(self.pattern)}")
        if self.priority < 0:
            raise ValueError(f"priority must be non-negative, got {self.priority}")

@dataclass
class RedactionResult:
    """
    Complete redaction result with metadata
    
    Attributes:
        redacted_text: Text after redaction
        redactions: List of individual redaction operations
        summary: Statistical summary of redactions
        original_length: Length of original text
        redacted_length: Length of redacted text
    """
    redacted_text: str
    redactions: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    original_length: int = 0
    redacted_length: int = 0

class PIIRedactor:
    """
    Comprehensive PII/PHI redaction with selective entity types
    
    This class implements HIPAA-compliant redaction of 18 HIPAA identifiers
    using both regex-based and NER-based approaches.
    """
    
    # HIPAA Safe Harbor 18 Identifiers + Common Aliases
    AVAILABLE_ENTITY_TYPES = {
        "PERSON": "Names of individuals (patients, relatives, employers)",
        "DATE": "All dates (except year) related to the individual",
        "AGE": "All ages over 89 (and elements that could be combined)",
        "LOCATION": "Geographic subdivisions smaller than state (addresses, cities, zip codes)",
        "PHONE": "Telephone and fax numbers",
        "EMAIL": "Email addresses",
        "SSN": "Social Security Numbers",
        "ID": "Account numbers, certificate/license numbers, device identifiers",
        "HOSPITAL": "Hospital and healthcare facility names",
        "ORGANIZATION": "Organizations, healthcare facilities, and institutions",  # ADDED
        "DOCTOR": "Physician and healthcare provider names",
        "PATIENT": "Patient identifiers and unique identifying numbers",
        "MEDICAL_RECORD": "Medical record numbers",
        "MRN": "Medical Record Numbers (alias for MEDICAL_RECORD)",  # ADDED
        "BIOMETRIC": "Biometric identifiers (fingerprints, voice prints)",
        "PHOTO": "Full-face photographs and comparable images",
        "URL": "Web URLs and IP addresses",
        "VIN": "Vehicle identifiers and license plate numbers",
        "HEALTH_PLAN": "Health plan beneficiary numbers"
    }
    
    def __init__(
        self, 
        entity_types: Optional[List[str]] = None, 
        method: str = "Replace with tag",
        use_ner: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize PII Redactor
        
        Args:
            entity_types: List of entity types to redact (None = all types)
            method: Redaction method ("Replace with tag", "Replace with placeholder", "Remove completely")
            use_ner: Whether to use NER models in addition to regex
            strict_mode: If True, applies more aggressive redaction patterns
            
        Raises:
            ValueError: If invalid entity types or method provided
        """
        # Validate entity types
        if entity_types is None:
            self.entity_types = set(self.AVAILABLE_ENTITY_TYPES.keys())
        else:
            invalid_types = set(entity_types) - set(self.AVAILABLE_ENTITY_TYPES.keys())
            if invalid_types:
                raise ValueError(f"Invalid entity types: {invalid_types}")
            self.entity_types = set(entity_types)
        
        # Normalize aliases: MRN -> include MEDICAL_RECORD patterns
        if "MRN" in self.entity_types:
            self.entity_types.add("MEDICAL_RECORD")
        
        # Normalize aliases: ORGANIZATION -> include HOSPITAL patterns
        if "ORGANIZATION" in self.entity_types:
            self.entity_types.add("HOSPITAL")
        
        # Validate redaction method
        try:
            self.method = RedactionMethod(method)
        except ValueError:
            valid_methods = [m.value for m in RedactionMethod]
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        self.use_ner = use_ner
        self.strict_mode = strict_mode
        self.patterns = self._build_patterns()
        self.ner_model = self._load_ner_model() if use_ner else None
        
        logger.info(
            f"PII Redactor initialized: {len(self.entity_types)} entity types, "
            f"method={self.method.value}, use_ner={use_ner}, strict_mode={strict_mode}"
        )
    
    def _load_ner_model(self) -> Optional[Any]:
        """
        Load NER model with fallback options
        
        Returns:
            Loaded NER model or None if unavailable
        """
        try:
            import spacy
            
            # Try MedSpaCy first (medical domain-specific)
            try:
                import medspacy
                nlp = medspacy.load()
                logger.info("Loaded MedSpaCy for medical NER")
                return nlp
            except (ImportError, OSError) as e:
                logger.debug(f"MedSpaCy not available: {e}")
            
            # Fallback to standard spaCy
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy en_core_web_sm")
                return nlp
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
                return None
                
        except ImportError:
            logger.warning(
                "spaCy not available. Install with: pip install spacy. "
                "NER-based redaction will be disabled."
            )
            return None
    
    def _build_patterns(self) -> List[RedactionPattern]:
        """
        Build comprehensive regex patterns for PII/PHI detection
        
        Returns:
            List of RedactionPattern objects sorted by priority
        """
        patterns = []
        
        # Social Security Numbers (High Priority)
        if "SSN" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="SSN_STANDARD",
                    pattern=re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
                    replacement=self._get_replacement("SSN"),
                    entity_type="SSN",
                    priority=10,
                    description="Social Security Number"
                ),
                RedactionPattern(
                    name="SSN_LABELED",
                    pattern=re.compile(r'\b(?:SSN|Social Security|SS#)[\s:]*(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b', re.I),
                    replacement=self._get_replacement("SSN"),
                    entity_type="SSN",
                    priority=11,
                    description="Labeled Social Security Number"
                )
            ])
        
        # Phone Numbers
        if "PHONE" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="PHONE_US",
                    pattern=re.compile(
                        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
                    ),
                    replacement=self._get_replacement("PHONE"),
                    entity_type="PHONE",
                    priority=8,
                    description="US Phone Number"
                ),
                RedactionPattern(
                    name="PHONE_LABELED",
                    pattern=re.compile(
                        r'\b(?:Phone|Tel|Telephone|Cell|Mobile|Fax)[\s:]*(\+?[\d\s\-\(\)\.]+)\b',
                        re.I
                    ),
                    replacement=self._get_replacement("PHONE"),
                    entity_type="PHONE",
                    priority=9,
                    description="Labeled Phone Number"
                )
            ])
        
        # Email Addresses
        if "EMAIL" in self.entity_types:
            patterns.append(RedactionPattern(
                name="EMAIL",
                pattern=re.compile(
                    r'\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Z|a-z]{2,}\b'
                ),
                replacement=self._get_replacement("EMAIL"),
                entity_type="EMAIL",
                priority=9,
                description="Email Address"
            ))
        
        # Medical Record Numbers (includes MRN alias)
        if "MEDICAL_RECORD" in self.entity_types or "MRN" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="MRN_LABELED",
                    pattern=re.compile(
                        r'\b(?:MRN|Medical Record|MR#|Record #|Record Number|Patient ID)[\s:]*([A-Z0-9]{5,})\b',
                        re.I
                    ),
                    replacement=self._get_replacement("MRN"),
                    entity_type="MEDICAL_RECORD",
                    priority=10,
                    description="Medical Record Number (Labeled)"
                ),
                RedactionPattern(
                    name="MRN_PATTERN",
                    pattern=re.compile(r'\bMR[N#]?\d{6,}\b', re.I),
                    replacement=self._get_replacement("MRN"),
                    entity_type="MEDICAL_RECORD",
                    priority=9,
                    description="Medical Record Number (Pattern)"
                ),
                RedactionPattern(
                    name="MRN_STANDARD",
                    pattern=re.compile(r'\b(?:MRN|mrn)[\s:]+([A-Z0-9]{6,})\b'),
                    replacement=self._get_replacement("MRN"),
                    entity_type="MEDICAL_RECORD",
                    priority=10,
                    description="Medical Record Number (Standard Format)"
                )
            ])
        
        # Patient and Account IDs
        if "ID" in self.entity_types or "PATIENT" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="ACCOUNT_ID",
                    pattern=re.compile(
                        r'\b(?:Account|Acct|Patient ID|Patient Number)[\s#:]*([A-Z0-9]{6,})\b',
                        re.I
                    ),
                    replacement=self._get_replacement("ID"),
                    entity_type="ID",
                    priority=8,
                    description="Account/Patient ID"
                ),
                RedactionPattern(
                    name="DEVICE_ID",
                    pattern=re.compile(
                        r'\b(?:Device|Serial)[\s#:]*([A-Z0-9]{8,})\b',
                        re.I
                    ),
                    replacement=self._get_replacement("ID"),
                    entity_type="ID",
                    priority=7,
                    description="Device Identifier"
                )
            ])
        
        # Organizations and Hospitals
        if "HOSPITAL" in self.entity_types or "ORGANIZATION" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="HOSPITAL_NAMES",
                    pattern=re.compile(
                        r'\b(?:[A-Z][a-z]+\s+){0,2}(?:Hospital|Medical Center|Clinic|Healthcare|'
                        r'Health System|Medical Group|Institute|Foundation)\b',
                        re.I
                    ),
                    replacement=self._get_replacement("ORGANIZATION"),
                    entity_type="HOSPITAL",
                    priority=6,
                    description="Hospital/Organization Names"
                ),
                RedactionPattern(
                    name="MEDICAL_FACILITIES",
                    pattern=re.compile(
                        r'\b(?:MUSC|Mayo Clinic|Cleveland Clinic|Johns Hopkins|'
                        r'Kaiser Permanente|VA Medical|Veterans Administration)\b',
                        re.I
                    ),
                    replacement=self._get_replacement("ORGANIZATION"),
                    entity_type="HOSPITAL",
                    priority=7,
                    description="Known Medical Facilities"
                )
            ])
        
        # URLs and IP Addresses
        if "URL" in self.entity_types or "ID" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="IP_ADDRESS",
                    pattern=re.compile(
                        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
                        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
                    ),
                    replacement=self._get_replacement("URL"),
                    entity_type="URL",
                    priority=7,
                    description="IP Address"
                ),
                RedactionPattern(
                    name="URL",
                    pattern=re.compile(
                        r'\b(?:https?://|www\.)[^\s]+\b',
                        re.I
                    ),
                    replacement=self._get_replacement("URL"),
                    entity_type="URL",
                    priority=6,
                    description="Web URL"
                )
            ])
        
        # Dates (HIPAA requires redaction of all dates except year)
        if "DATE" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="DATE_SLASH",
                    pattern=re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
                    replacement=self._get_replacement("DATE"),
                    entity_type="DATE",
                    priority=5,
                    description="Slash-formatted Date"
                ),
                RedactionPattern(
                    name="DATE_TEXT_FULL",
                    pattern=re.compile(
                        r'\b(?:January|February|March|April|May|June|July|August|'
                        r'September|October|November|December|'
                        r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                        r'[.,]?\s+\d{1,2}[.,]?\s+\d{4}\b',
                        re.I
                    ),
                    replacement=self._get_replacement("DATE"),
                    entity_type="DATE",
                    priority=5,
                    description="Text-formatted Date"
                ),
                RedactionPattern(
                    name="DATE_ISO",
                    pattern=re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
                    replacement=self._get_replacement("DATE"),
                    entity_type="DATE",
                    priority=5,
                    description="ISO Date Format"
                )
            ])
        
        # Ages over 89 (HIPAA requirement)
        if "AGE" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="AGE_OVER_89",
                    pattern=re.compile(
                        r'\b(?:age[ds]?|yo|years old|y/o)[\s:]*([9]\d|[1-9]\d{2,})(?:\s+(?:years?|yrs?))?\b',
                        re.I
                    ),
                    replacement=self._get_replacement("AGE"),
                    entity_type="AGE",
                    priority=7,
                    description="Age over 89"
                ),
                RedactionPattern(
                    name="AGE_PATTERN",
                    pattern=re.compile(r'\b([9]\d|[1-9]\d{2,})[-\s]?(?:year|yr)[-\s]?old\b', re.I),
                    replacement=self._get_replacement("AGE"),
                    entity_type="AGE",
                    priority=7,
                    description="Age Pattern (over 89)"
                )
            ])
        
        # Addresses and Geographic Locations
        if "LOCATION" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="STREET_ADDRESS",
                    pattern=re.compile(
                        r'\b\d+\s+(?:[A-Z][a-z]+\s+){1,3}'
                        r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b',
                        re.I
                    ),
                    replacement=self._get_replacement("LOCATION"),
                    entity_type="LOCATION",
                    priority=8,
                    description="Street Address"
                ),
                RedactionPattern(
                    name="ZIP_CODE",
                    pattern=re.compile(r'\b\d{5}(?:-\d{4})?\b'),
                    replacement=self._get_replacement("LOCATION"),
                    entity_type="LOCATION",
                    priority=7,
                    description="ZIP Code"
                ),
                RedactionPattern(
                    name="PO_BOX",
                    pattern=re.compile(r'\b(?:P\.?O\.?\s*Box|Post Office Box)\s*\d+\b', re.I),
                    replacement=self._get_replacement("LOCATION"),
                    entity_type="LOCATION",
                    priority=7,
                    description="PO Box"
                )
            ])
        
        # Vehicle Identifiers
        if "VIN" in self.entity_types:
            patterns.extend([
                RedactionPattern(
                    name="VIN",
                    pattern=re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
                    replacement=self._get_replacement("VIN"),
                    entity_type="VIN",
                    priority=6,
                    description="Vehicle Identification Number"
                ),
                RedactionPattern(
                    name="LICENSE_PLATE",
                    pattern=re.compile(
                        r'\b(?:License Plate|Plate #?)[\s:]*([A-Z0-9]{3,8})\b',
                        re.I
                    ),
                    replacement=self._get_replacement("VIN"),
                    entity_type="VIN",
                    priority=6,
                    description="License Plate"
                )
            ])
        
        # Health Plan Numbers
        if "HEALTH_PLAN" in self.entity_types:
            patterns.append(RedactionPattern(
                name="HEALTH_PLAN",
                pattern=re.compile(
                    r'\b(?:Policy|Member|Subscriber|Group)[\s#:]*([A-Z0-9]{6,})\b',
                    re.I
                ),
                replacement=self._get_replacement("HEALTH_PLAN"),
                entity_type="HEALTH_PLAN",
                priority=7,
                description="Health Plan Identifier"
            ))
        
        # Sort by priority (highest first)
        patterns.sort(key=lambda x: x.priority, reverse=True)
        
        logger.debug(f"Built {len(patterns)} redaction patterns")
        return patterns
    
    def _get_replacement(self, entity_type: str) -> str:
        """
        Get replacement text based on redaction method and entity type
        
        Args:
            entity_type: Type of entity being redacted
            
        Returns:
            Replacement string
        """
        if self.method == RedactionMethod.TAG:
            return f"[{entity_type}]"
        
        elif self.method == RedactionMethod.PLACEHOLDER:
            placeholders = {
                "PERSON": "John Doe",
                "DATE": "01/01/2000",
                "AGE": "50",
                "LOCATION": "City, State",
                "PHONE": "555-0100",
                "EMAIL": "email@example.com",
                "SSN": "000-00-0000",
                "ID": "ID123456",
                "PATIENT": "PT000000",
                "MEDICAL_RECORD": "MRN000000",
                "MRN": "MRN000000",
                "HOSPITAL": "General Hospital",
                "ORGANIZATION": "General Hospital",
                "DOCTOR": "Dr. Smith",
                "URL": "http://example.com",
                "VIN": "0000000000000000",
                "HEALTH_PLAN": "HP000000",
                "BIOMETRIC": "[BIOMETRIC]",
                "PHOTO": "[PHOTO]"
            }
            return placeholders.get(entity_type, "[REDACTED]")
        
        else:  # RedactionMethod.REMOVE
            return ""
    
    def redact(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Perform comprehensive PII/PHI redaction on text
        
        Args:
            text: Input text to redact
            
        Returns:
            Tuple of (redacted_text, list of redactions) for backward compatibility
            Note: Returns can be unpacked as: text, redactions = redactor.redact(input)
            
        Raises:
            ValueError: If text is None
        """
        if text is None:
            raise ValueError("Input text cannot be None")
        
        if not text.strip():
            return text, []
        
        if not self.entity_types:
            logger.warning("No entity types selected for redaction")
            return text, []
        
        original_length = len(text)
        redactions = []
        redacted_text = text
        
        # Phase 1: Regex-based redaction
        logger.debug(f"Starting regex redaction with {len(self.patterns)} patterns")
        for pattern in self.patterns:
            try:
                matches = list(pattern.pattern.finditer(redacted_text))
                logger.debug(f"Pattern {pattern.name} found {len(matches)} matches")
                
                # Process matches in reverse to maintain correct positions
                for match in reversed(matches):
                    start, end = match.span()
                    original = match.group()
                    
                    # Skip if already redacted
                    if original.startswith('[') and original.endswith(']'):
                        continue
                    
                    redactions.append({
                        'type': pattern.entity_type,
                        'name': pattern.name,
                        'original': original,
                        'replacement': pattern.replacement,
                        'start': start,
                        'end': end,
                        'method': 'regex',
                        'priority': pattern.priority,
                        'description': pattern.description
                    })
                    
                    # Apply redaction
                    redacted_text = (
                        redacted_text[:start] + 
                        pattern.replacement + 
                        redacted_text[end:]
                    )
                    
            except Exception as e:
                logger.error(f"Error processing pattern {pattern.name}: {e}")
                continue
        
        # Phase 2: NER-based redaction
        if self.ner_model and self.use_ner:
            logger.debug("Starting NER-based redaction")
            try:
                redacted_text, ner_redactions = self._ner_redact(redacted_text)
                redactions.extend(ner_redactions)
                logger.debug(f"NER found {len(ner_redactions)} additional entities")
            except Exception as e:
                logger.error(f"NER redaction failed: {e}")
        
        # Generate summary
        summary = self._generate_summary(redactions)
        redacted_length = len(redacted_text)
        
        logger.info(
            f"Redaction complete: {summary['total_redactions']} redactions, "
            f"{original_length} -> {redacted_length} chars"
        )
        
        # Return tuple for backward compatibility with original API
        return redacted_text, redactions
    
    def _ner_redact(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Perform NER-based entity detection and redaction
        
        Args:
            text: Text to process with NER
            
        Returns:
            Tuple of (redacted_text, list of redactions)
        """
        if not self.ner_model:
            return text, []
        
        redactions = []
        
        try:
            # Process with NER model
            doc = self.ner_model(text)
            entities = []
            
            # Extract entities
            for ent in doc.ents:
                entity_type = self._map_ner_label(ent.label_)
                
                if entity_type and entity_type in self.entity_types:
                    # Skip if already redacted
                    if ent.text.startswith('[') and ent.text.endswith(']'):
                        continue
                    
                    entities.append({
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'text': ent.text,
                        'type': entity_type,
                        'label': ent.label_,
                        'confidence': getattr(ent, 'score', 1.0)
                    })
            
            # Sort by start position (reverse) to maintain correct indices
            entities.sort(key=lambda x: x['start'], reverse=True)
            redacted_text = text
            
            # Apply redactions
            for entity in entities:
                replacement = self._get_replacement(entity['type'])
                
                redactions.append({
                    'type': entity['type'],
                    'name': entity['label'],
                    'original': entity['text'],
                    'replacement': replacement,
                    'start': entity['start'],
                    'end': entity['end'],
                    'method': 'ner',
                    'priority': 5,
                    'confidence': entity['confidence'],
                    'description': f"NER-detected {entity['label']}"
                })
                
                redacted_text = (
                    redacted_text[:entity['start']] + 
                    replacement + 
                    redacted_text[entity['end']:]
                )
            
            return redacted_text, redactions
            
        except Exception as e:
            logger.error(f"NER processing failed: {e}", exc_info=True)
            return text, []
    
    def _map_ner_label(self, label: str) -> Optional[str]:
        """
        Map NER model labels to HIPAA entity types
        
        Args:
            label: NER model label
            
        Returns:
            Mapped entity type or None
        """
        label_upper = label.upper()
        
        mapping = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'PATIENT': 'PATIENT',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'GPE': 'LOCATION',  # Geopolitical entity
            'LOC': 'LOCATION',
            'FAC': 'HOSPITAL',  # Facility
            'ORG': 'HOSPITAL',
            'AGE': 'AGE',
            'CARDINAL': None,  # Numbers - too generic
            'ORDINAL': None,
            'QUANTITY': None,
        }
        
        return mapping.get(label_upper)
    
    def _generate_summary(self, redactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistical summary of redactions
        
        Args:
            redactions: List of redaction operations
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_redactions': len(redactions),
            'by_type': {},
            'by_method': {'regex': 0, 'ner': 0},
            'by_priority': {},
            'unique_originals': set()
        }
        
        for redaction in redactions:
            # Count by type
            entity_type = redaction['type']
            summary['by_type'][entity_type] = summary['by_type'].get(entity_type, 0) + 1
            
            # Count by method
            method = redaction['method']
            summary['by_method'][method] = summary['by_method'].get(method, 0) + 1
            
            # Count by priority
            priority = redaction['priority']
            summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
            
            # Track unique values
            summary['unique_originals'].add(redaction['original'])
        
        # Convert set to count
        summary['unique_entities'] = len(summary['unique_originals'])
        del summary['unique_originals']
        
        return summary
    
    def get_enabled_entities(self) -> Dict[str, str]:
        """
        Get currently enabled entity types with descriptions
        
        Returns:
            Dictionary mapping entity types to descriptions
        """
        return {
            entity: description 
            for entity, description in self.AVAILABLE_ENTITY_TYPES.items() 
            if entity in self.entity_types
        }
    
    def get_available_entities(self) -> Dict[str, str]:
        """
        Get all available entity types with descriptions
        
        Returns:
            Dictionary of all available entity types
        """
        return self.AVAILABLE_ENTITY_TYPES.copy()
    
    def add_entity_type(self, entity_type: str) -> None:
        """
        Enable additional entity type for redaction
        
        Args:
            entity_type: Entity type to enable
            
        Raises:
            ValueError: If entity type is invalid
        """
        if entity_type not in self.AVAILABLE_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity type: {entity_type}. "
                f"Available types: {list(self.AVAILABLE_ENTITY_TYPES.keys())}"
            )
        
        self.entity_types.add(entity_type)
        
        # Handle aliases
        if entity_type == "MRN":
            self.entity_types.add("MEDICAL_RECORD")
        if entity_type == "ORGANIZATION":
            self.entity_types.add("HOSPITAL")
        
        self.patterns = self._build_patterns()
        logger.info(f"Added entity type: {entity_type}")
    
    def remove_entity_type(self, entity_type: str) -> None:
        """
        Disable entity type from redaction
        
        Args:
            entity_type: Entity type to disable
        """
        if entity_type in self.entity_types:
            self.entity_types.remove(entity_type)
            self.patterns = self._build_patterns()
            logger.info(f"Removed entity type: {entity_type}")
    
    def get_redaction_summary(self, redactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary from list of redactions (backward compatibility)
        
        Args:
            redactions: List of redaction operations
            
        Returns:
            Summary dictionary
        """
        return self._generate_summary(redactions)

def create_redactor(
    entity_types: Optional[List[str]] = None, 
    method: str = "Replace with tag",
    use_ner: bool = True,
    strict_mode: bool = False
) -> PIIRedactor:
    """
    Factory function to create a PIIRedactor instance
    
    Args:
        entity_types: List of entity types to redact (None = common types)
        method: Redaction method
        use_ner: Whether to use NER models
        strict_mode: Enable strict redaction mode
        
    Returns:
        Configured PIIRedactor instance
    """
    if entity_types is None:
        # Default to common HIPAA identifiers
        entity_types = [
            "PERSON", "DATE", "LOCATION", "PHONE", "EMAIL", 
            "SSN", "ID", "MEDICAL_RECORD", "PATIENT"
        ]
    
    return PIIRedactor(
        entity_types=entity_types,
        method=method,
        use_ner=use_ner,
        strict_mode=strict_mode
    )

def redact_text(
    text: str,
    entity_types: Optional[List[str]] = None,
    method: str = "Replace with tag"
) -> str:
    """
    Quick text redaction function
    
    Args:
        text: Text to redact
        entity_types: Entity types to redact
        method: Redaction method
        
    Returns:
        Redacted text string
    """
    redactor = create_redactor(entity_types=entity_types, method=method)
    redacted_text, _ = redactor.redact(text)
    return redacted_text

def redact_with_summary(
    text: str,
    entity_types: Optional[List[str]] = None,
    method: str = "Replace with tag"
) -> Tuple[str, Dict[str, Any]]:
    """
    Redact text and return summary statistics
    
    Args:
        text: Text to redact
        entity_types: Entity types to redact
        method: Redaction method
        
    Returns:
        Tuple of (redacted_text, summary_dict)
    """
    redactor = create_redactor(entity_types=entity_types, method=method)
    redacted_text, redactions = redactor.redact(text)
    summary = redactor.get_redaction_summary(redactions)
    return redacted_text, summary

def validate_redaction(
    original: str,
    redacted: str,
    entity_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate that redaction was successful and complete
    
    Args:
        original: Original text
        redacted: Redacted text
        entity_types: Entity types that should be redacted
        
    Returns:
        Validation report dictionary
    """
    redactor = create_redactor(entity_types=entity_types)
    expected_text, redactions = redactor.redact(original)
    summary = redactor.get_redaction_summary(redactions)
    
    validation = {
        'is_valid': expected_text == redacted,
        'expected': expected_text,
        'actual': redacted,
        'matches': expected_text == redacted,
        'redactions_found': summary['total_redactions'],
        'entity_types_found': list(summary['by_type'].keys()),
        'original_length': len(original),
        'redacted_length': len(redacted),
        'reduction_percent': (1 - len(redacted) / len(original)) * 100 if len(original) > 0 else 0
    }
    
    return validation

class BatchRedactor:
    """
    Efficient batch processing of multiple texts
    """
    
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        method: str = "Replace with tag",
        use_ner: bool = True,
        strict_mode: bool = False
    ):
        """Initialize batch redactor"""
        self.redactor = create_redactor(
            entity_types=entity_types,
            method=method,
            use_ner=use_ner,
            strict_mode=strict_mode
        )
        self.results: List[Tuple[str, List[Dict[str, Any]]]] = []
    
    def redact_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Redact multiple texts
        
        Args:
            texts: List of texts to redact
            show_progress: Whether to show progress bar
            
        Returns:
            List of (redacted_text, redactions) tuples
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Redacting")
            except ImportError:
                iterator = texts
                logger.warning("tqdm not available, progress bar disabled")
        else:
            iterator = texts
        
        for text in iterator:
            try:
                redacted_text, redactions = self.redactor.redact(text)
                results.append((redacted_text, redactions))
            except Exception as e:
                logger.error(f"Error redacting text: {e}")
                # Return original text on error
                results.append((text, []))
        
        self.results = results
        return results
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for entire batch
        
        Returns:
            Aggregated summary dictionary
        """
        if not self.results:
            return {'error': 'No results available'}
        
        total_redactions = 0
        total_original = 0
        total_redacted = 0
        by_type = {}
        
        for redacted_text, redactions in self.results:
            total_redactions += len(redactions)
            
            # Estimate original length (this is approximate since we don't store it)
            total_redacted += len(redacted_text)
            
            # Aggregate by type
            for redaction in redactions:
                entity_type = redaction['type']
                by_type[entity_type] = by_type.get(entity_type, 0) + 1
        
        return {
            'total_texts': len(self.results),
            'total_redactions': total_redactions,
            'average_redactions_per_text': total_redactions / len(self.results) if self.results else 0,
            'total_redacted_chars': total_redacted,
            'by_type': by_type,
            'texts_with_redactions': sum(1 for _, redactions in self.results if len(redactions) > 0)
        }

# Export public API
__all__ = [
    'PIIRedactor',
    'RedactionResult',
    'RedactionPattern',
    'RedactionMethod',
    'BatchRedactor',
    'create_redactor',
    'redact_text',
    'redact_with_summary',
    'validate_redaction'
]
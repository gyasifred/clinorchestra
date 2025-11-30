#!/usr/bin/env python3
"""
Remove all emojis from classification prompts for professional publication
Replaces emojis with appropriate text equivalents
"""

import re
from pathlib import Path

# Define emoji replacements
EMOJI_REPLACEMENTS = {
    # Check/cross marks
    '✅': '',  # Remove, text already says what to do
    '❌': '',  # Remove, "DO NOT" or "CRITICAL" already present
    '☐': '[ ]',  # Checkbox unchecked
    '☑': '[x]',  # Checkbox checked
    '✓': '[x]',  # Check mark
    '✗': '[ ]',  # X mark

    # Other common emojis
    '⚠️': 'WARNING:',
    '🔍': '',
    '📋': '',
    '⭐': '',
    '💡': 'NOTE:',
    '🎯': '',
    '📌': 'IMPORTANT:',
    '🚫': 'PROHIBITED:',
    '⚡': '',
    '🔥': '',
}

def remove_emojis(text: str) -> str:
    """Remove emojis and replace with text equivalents"""

    # First pass: Known emoji replacements
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        if emoji in text:
            # If replacement is empty and emoji is at start of line after "- "
            # just remove the emoji
            if replacement == '':
                text = text.replace(f'- {emoji} ', '- ')
                text = text.replace(emoji, '')
            else:
                text = text.replace(emoji, replacement)

    # Second pass: Remove any remaining emojis using Unicode ranges
    # This catches any emojis we might have missed
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002700-\U000027BF"  # dingbats
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Clean up any double spaces created by emoji removal
    text = re.sub(r'  +', ' ', text)

    # Clean up lines that start with just "- " after emoji removal
    text = re.sub(r'^- \n', '- ', text, flags=re.MULTILINE)

    return text

def process_file(file_path: Path):
    """Process a single prompt file"""
    print(f"Processing: {file_path}")

    # Read original content
    with open(file_path, 'r', encoding='utf-8') as f:
        original = f.read()

    # Remove emojis
    cleaned = remove_emojis(original)

    # Count changes
    original_emojis = len([c for c in original if c in EMOJI_REPLACEMENTS.keys()])

    # Write cleaned content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"  Removed {original_emojis} known emojis")
    print(f"  File updated: {file_path}")

def main():
    """Process all classification prompt files"""

    # Define files to process
    prompt_files = [
        'examples/adrd_classification/prompts/main_prompt.txt',
        'examples/adrd_classification/prompts/rag_refinement_prompt.txt',
        'examples/malnutrition_classification_only/main_prompt.txt',
        'examples/malnutrition_classification_only/refinement_prompt.txt',
    ]

    base_dir = Path('/home/user/clinorchestra')

    print("=" * 60)
    print("REMOVING EMOJIS FROM CLASSIFICATION PROMPTS")
    print("=" * 60)
    print()

    for file_rel_path in prompt_files:
        file_path = base_dir / file_rel_path

        if not file_path.exists():
            print(f"WARNING: File not found: {file_path}")
            continue

        process_file(file_path)
        print()

    print("=" * 60)
    print("EMOJI REMOVAL COMPLETE")
    print("All prompts are now emoji-free and publication-ready")
    print("=" * 60)

if __name__ == '__main__':
    main()

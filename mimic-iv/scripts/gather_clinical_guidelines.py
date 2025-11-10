"""
Clinical Guidelines Gathering Assistant

This script helps you gather clinical guidelines for the top 20 diagnoses
from PubMed, clinical societies, and other reputable sources.

Note: This script provides search queries and URLs. You'll need to manually
download the PDFs/documents due to access restrictions.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class GuidelineGatherer:
    """Helper class to generate search queries and organize guidelines"""

    def __init__(self, top_diagnoses_path: str):
        """Initialize with top diagnoses file"""
        self.top_diagnoses = pd.read_csv(top_diagnoses_path)
        self.guidelines_dir = Path("mimic-iv/guidelines")
        self.guidelines_dir.mkdir(exist_ok=True)

        # Clinical societies and guideline sources
        self.guideline_sources = {
            "General Medical": [
                "UpToDate",
                "DynaMed",
                "BMJ Best Practice"
            ],
            "Cardiology": [
                "American Heart Association (AHA)",
                "American College of Cardiology (ACC)",
                "European Society of Cardiology (ESC)"
            ],
            "Pulmonary": [
                "American Thoracic Society (ATS)",
                "European Respiratory Society (ERS)",
                "GOLD (COPD guidelines)"
            ],
            "Infectious Disease": [
                "Infectious Diseases Society of America (IDSA)",
                "Surviving Sepsis Campaign",
                "CDC Guidelines"
            ],
            "Nephrology": [
                "Kidney Disease: Improving Global Outcomes (KDIGO)",
                "National Kidney Foundation"
            ],
            "Critical Care": [
                "Society of Critical Care Medicine (SCCM)",
                "European Society of Intensive Care Medicine (ESICM)"
            ],
            "Gastroenterology": [
                "American Gastroenterological Association (AGA)",
                "American College of Gastroenterology (ACG)"
            ],
            "Neurology": [
                "American Academy of Neurology (AAN)",
                "American Stroke Association (ASA)"
            ]
        }

    def generate_search_queries(self, diagnosis_name: str, icd_code: str):
        """Generate search queries for a diagnosis"""

        queries = {
            "pubmed": [
                f"{diagnosis_name} clinical practice guidelines",
                f"{diagnosis_name} diagnosis criteria",
                f"{diagnosis_name} treatment guidelines",
                f"ICD {icd_code} clinical guidelines",
                f"{diagnosis_name} systematic review meta-analysis"
            ],
            "google_scholar": [
                f"{diagnosis_name} guidelines 2020-2024",
                f"{diagnosis_name} diagnostic criteria evidence based"
            ],
            "guidelines_databases": [
                f"National Guideline Clearinghouse {diagnosis_name}",
                f"NICE guidelines {diagnosis_name}",
                f"WHO guidelines {diagnosis_name}"
            ]
        }

        return queries

    def generate_pubmed_urls(self, diagnosis_name: str):
        """Generate PubMed search URLs"""

        base_url = "https://pubmed.ncbi.nlm.nih.gov/"

        searches = []

        # Clinical practice guidelines
        query = f'("{diagnosis_name}"[Title/Abstract]) AND ("clinical practice guideline"[Publication Type] OR "guideline"[Publication Type])'
        searches.append({
            "name": "Clinical Practice Guidelines",
            "url": f"{base_url}?term={query.replace(' ', '+')}&filter=pubt.guideline&filter=years.2018-2024"
        })

        # Systematic reviews
        query = f'("{diagnosis_name}"[Title/Abstract]) AND ("systematic review"[Publication Type] OR "meta-analysis"[Publication Type])'
        searches.append({
            "name": "Systematic Reviews & Meta-analyses",
            "url": f"{base_url}?term={query.replace(' ', '+')}&filter=pubt.systematicreview&filter=years.2018-2024"
        })

        # Diagnostic criteria
        query = f'("{diagnosis_name}"[Title/Abstract]) AND ("diagnosis"[Title/Abstract] OR "diagnostic criteria"[Title/Abstract])'
        searches.append({
            "name": "Diagnostic Criteria",
            "url": f"{base_url}?term={query.replace(' ', '+')}&filter=years.2018-2024"
        })

        return searches

    def create_guideline_collection_guide(self):
        """Create a comprehensive guide for collecting guidelines"""

        print("="*80)
        print("CLINICAL GUIDELINES COLLECTION GUIDE")
        print("="*80)
        print(f"\nFound {len(self.top_diagnoses)} diagnoses to collect guidelines for\n")

        all_guides = []

        for idx, row in self.top_diagnoses.iterrows():
            diagnosis_name = row['long_title']
            icd_code = row['icd_code']
            icd_version = row['icd_version']

            print(f"\n{'='*80}")
            print(f"{idx+1}. {diagnosis_name} (ICD-{icd_version}: {icd_code})")
            print(f"{'='*80}")

            # Create directory for this diagnosis
            diagnosis_dir = self.guidelines_dir / f"{idx+1:02d}_{icd_code.replace('.', '_')}"
            diagnosis_dir.mkdir(exist_ok=True)

            # Generate search queries
            queries = self.generate_search_queries(diagnosis_name, icd_code)

            print("\n📚 PUBMED SEARCHES:")
            pubmed_urls = self.generate_pubmed_urls(diagnosis_name)
            for search in pubmed_urls:
                print(f"\n  {search['name']}:")
                print(f"  {search['url']}")

            print("\n🔍 SEARCH QUERIES:")
            for source, query_list in queries.items():
                print(f"\n  {source.upper()}:")
                for q in query_list:
                    print(f"    - {q}")

            print(f"\n📁 Save guidelines to: {diagnosis_dir}/")

            # Create a JSON file with search info
            guide_info = {
                "diagnosis": diagnosis_name,
                "icd_code": icd_code,
                "icd_version": int(icd_version),
                "directory": str(diagnosis_dir),
                "search_queries": queries,
                "pubmed_searches": pubmed_urls,
                "recommended_sources": self._get_recommended_sources_for_diagnosis(diagnosis_name),
                "collection_checklist": [
                    "Clinical practice guidelines from major societies",
                    "Diagnostic criteria (established scoring systems)",
                    "Evidence-based treatment protocols",
                    "Recent systematic reviews (last 5 years)",
                    "Key research papers from high-impact journals"
                ],
                "file_naming_convention": [
                    "guideline_[organization]_[year].pdf",
                    "diagnostic_criteria_[name].pdf",
                    "systematic_review_[firstauthor]_[year].pdf"
                ]
            }

            # Save guide
            guide_path = diagnosis_dir / "collection_guide.json"
            with open(guide_path, 'w', encoding='utf-8') as f:
                json.dump(guide_info, f, indent=2)

            all_guides.append(guide_info)

            print(f"  ✓ Collection guide saved to: {guide_path}")

        # Save master index
        master_index = {
            "created_date": datetime.now().isoformat(),
            "total_diagnoses": len(self.top_diagnoses),
            "diagnoses": all_guides
        }

        master_path = self.guidelines_dir / "master_index.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_index, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✓ Master index saved to: {master_path}")
        print(f"{'='*80}")

        # Create a README for the guidelines directory
        self._create_guidelines_readme()

        return all_guides

    def _get_recommended_sources_for_diagnosis(self, diagnosis_name: str):
        """Get recommended guideline sources based on diagnosis category"""

        # This is a simplified categorization - you may want to make it more sophisticated
        diagnosis_lower = diagnosis_name.lower()

        recommendations = []

        if any(word in diagnosis_lower for word in ['heart', 'cardiac', 'myocardial', 'coronary']):
            recommendations.extend([
                "American Heart Association (AHA) - https://www.heart.org/",
                "American College of Cardiology (ACC) - https://www.acc.org/",
                "European Society of Cardiology - https://www.escardio.org/"
            ])

        if any(word in diagnosis_lower for word in ['pneumonia', 'respiratory', 'copd', 'asthma', 'lung']):
            recommendations.extend([
                "American Thoracic Society - https://www.thoracic.org/",
                "European Respiratory Society - https://www.ersnet.org/"
            ])

        if any(word in diagnosis_lower for word in ['sepsis', 'infection', 'pneumonia']):
            recommendations.extend([
                "Infectious Diseases Society of America - https://www.idsociety.org/",
                "Surviving Sepsis Campaign - https://www.sccm.org/SurvivingSepsisCampaign/"
            ])

        if any(word in diagnosis_lower for word in ['kidney', 'renal', 'dialysis']):
            recommendations.extend([
                "KDIGO - https://kdigo.org/",
                "National Kidney Foundation - https://www.kidney.org/"
            ])

        if any(word in diagnosis_lower for word in ['stroke', 'cerebral', 'brain']):
            recommendations.extend([
                "American Stroke Association - https://www.stroke.org/",
                "American Academy of Neurology - https://www.aan.com/"
            ])

        # Always include general sources
        recommendations.extend([
            "UpToDate - https://www.uptodate.com/",
            "BMJ Best Practice - https://bestpractice.bmj.com/",
            "PubMed - https://pubmed.ncbi.nlm.nih.gov/"
        ])

        return recommendations

    def _create_guidelines_readme(self):
        """Create README for guidelines directory"""

        readme_content = """# Clinical Guidelines Repository

This directory contains clinical practice guidelines, diagnostic criteria, and reference materials for the top 20 primary diagnoses in the MIMIC-IV dataset.

## Directory Structure

Each diagnosis has its own subdirectory with:
- `collection_guide.json` - Search queries and recommended sources
- Clinical practice guidelines (PDFs)
- Diagnostic criteria documents
- Systematic reviews
- Key research papers

## Naming Convention

Files should be named according to:
- `guideline_[organization]_[year].pdf` - e.g., `guideline_AHA_2021.pdf`
- `diagnostic_criteria_[name].pdf` - e.g., `diagnostic_criteria_SOFA_score.pdf`
- `systematic_review_[firstauthor]_[year].pdf` - e.g., `systematic_review_smith_2023.pdf`

## How to Use

1. Navigate to each diagnosis directory
2. Review the `collection_guide.json` for search queries
3. Download guidelines from recommended sources
4. Save files using the naming convention
5. These guidelines will be used as RAG knowledge base in ClinOrchestra

## Quality Criteria

Prioritize:
- ✓ Recent guidelines (last 5 years preferred)
- ✓ Evidence-based (systematic reviews, meta-analyses)
- ✓ From reputable organizations (AHA, ATS, IDSA, etc.)
- ✓ Peer-reviewed publications
- ✓ High-impact journals (NEJM, JAMA, Lancet, etc.)

## Copyright Notice

Clinical guidelines are copyrighted by their respective organizations.
Use for research and educational purposes only, following fair use principles.

## Using Guidelines in ClinOrchestra

Once collected:
1. In ClinOrchestra UI, go to RAG tab
2. Upload PDFs for relevant diagnoses
3. The system will create embeddings for semantic search
4. During annotation, relevant guidelines will be retrieved automatically

---

Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""

        readme_path = self.guidelines_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✓ Guidelines README created: {readme_path}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CLINICAL GUIDELINES GATHERING ASSISTANT")
    print("="*80)

    top_diagnoses_path = input("\nPath to top_20_primary_diagnoses.csv [mimic-iv/top_20_primary_diagnoses.csv]: ").strip()
    if not top_diagnoses_path:
        top_diagnoses_path = "mimic-iv/top_20_primary_diagnoses.csv"

    try:
        gatherer = GuidelineGatherer(top_diagnoses_path)
        guides = gatherer.create_guideline_collection_guide()

        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Review the search URLs and queries generated above")
        print("2. For each diagnosis, visit the PubMed URLs")
        print("3. Download relevant guidelines and save to the corresponding directory")
        print("4. Follow the file naming convention specified")
        print("5. Once collected, upload to ClinOrchestra RAG system")
        print("\nTIP: Focus on the most common diagnoses first (top 5-10)")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you've run extract_top_diagnoses.py first!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

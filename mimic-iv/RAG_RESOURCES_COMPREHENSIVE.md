# Comprehensive RAG Resources for 20 MIMIC-IV Diagnoses

This document provides curated, authoritative clinical resources for all 20 consolidated diagnoses. Use these as RAG (Retrieval-Augmented Generation) sources in ClinOrchestra to enhance AI-generated clinical annotations.

## How to Use

1. **Download PDFs**: Click links to download guidelines to your local machine
2. **Add to ClinOrchestra RAG**: In the RAG tab, upload PDFs or provide URLs
3. **Configure**: Set chunk_size=512, k_value=3 for optimal retrieval
4. **Process**: RAG will automatically retrieve relevant sections during annotation

---

## CARDIOVASCULAR DIAGNOSES

### 1. Chest Pain

**Primary Guideline:**
- **AHA/ACC Chest Pain Guideline (2021)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001029
  - Title: "2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain"
  - Citation: Gulati M, et al. Circulation. 2021;144:e368-e454

**Risk Stratification:**
- **HEART Score Validation**
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/23415740/
  - PMID: 23415740
  - Backus BE, et al. Int J Cardiol. 2013

- **TIMI Risk Score**
  - Original paper: https://pubmed.ncbi.nlm.nih.gov/10938172/
  - PMID: 10938172
  - Antman EM, et al. JAMA. 2000

### 2. Coronary Atherosclerosis

**Primary Guidelines:**
- **AHA/ACC Guideline on Management of Patients with Chronic Coronary Disease (2023)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001168
  - Virani SS, et al. Circulation. 2023;148:e9-e119

- **ESC Guidelines on Chronic Coronary Syndromes (2019)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/41/3/407/33531866/ehz425.pdf
  - Knuuti J, et al. Eur Heart J. 2020;41:407-477

**SYNTAX Score Calculator:**
- Resource: http://www.syntaxscore.org/
- For PCI vs CABG decision-making

### 3. Myocardial Infarction (MI)

**Universal Definition:**
- **Fourth Universal Definition of Myocardial Infarction (2018)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000000617
  - Thygesen K, et al. Circulation. 2018;138:e618-e651
  - **CRITICAL** - Defines MI types, biomarker criteria, ECG criteria

**STEMI Management:**
- **AHA/ACC STEMI Guideline (2013, Updated 2015)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000000336
  - O'Gara PT, et al. Circulation. 2013;127:e362-e425

**NSTEMI/UA Management:**
- **AHA/ACC NSTEMI Guideline (2014)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000000134
  - Amsterdam EA, et al. Circulation. 2014;130:e344-e426

### 4. Heart Failure

**Primary Guideline:**
- **AHA/ACC/HFSA Heart Failure Guideline (2022)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001063
  - Heidenreich PA, et al. Circulation. 2022;145:e895-e1032
  - **Includes GDMT, SGLT2i, ARNI recommendations**

- **ESC Heart Failure Guidelines (2021)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/42/36/3599/40403357/ehab368.pdf
  - McDonagh TA, et al. Eur Heart J. 2021;42:3599-3726

**Framingham Criteria:**
- Classic diagnostic criteria: https://pubmed.ncbi.nlm.nih.gov/3901634/
- PMID: 3901634
- McKee PA, et al. Circulation. 1971

### 5. Atrial Fibrillation

**Primary Guidelines:**
- **AHA/ACC/HRS AFib Guideline (2023)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000001193
  - Joglar JA, et al. Circulation. 2024;149:e1-e156

- **ESC AFib Guidelines (2020)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/42/5/373/36387486/ehaa612.pdf
  - Hindricks G, et al. Eur Heart J. 2021;42:373-498

**Risk Scores:**
- **CHA2DS2-VASc Score**
  - Validation: https://pubmed.ncbi.nlm.nih.gov/20299623/
  - PMID: 20299623
  - Lip GYH, et al. Chest. 2010

- **HAS-BLED Score**
  - Original: https://pubmed.ncbi.nlm.nih.gov/19762550/
  - PMID: 19762550
  - Pisters R, et al. Chest. 2010

### 6. Hypertensive Heart Disease with CKD

**Hypertension Guidelines:**
- **ACC/AHA Hypertension Guideline (2017)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/HYP.0000000000000065
  - Whelton PK, et al. Hypertension. 2018;71:e13-e115

**CKD in Hypertension:**
- **KDIGO Clinical Practice Guideline for CKD (2024)**
  - Direct PDF: https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf
  - Kidney Int. 2024;105(4S):S117-S314

- **Cardiorenal Syndromes:**
  - Review: https://pubmed.ncbi.nlm.nih.gov/18848134/
  - PMID: 18848134
  - Ronco C, et al. J Am Coll Cardiol. 2008

### 14. Syncope

**Primary Guideline:**
- **ESC Syncope Guidelines (2018)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/39/21/1883/27833945/ehy037.pdf
  - Brignole M, et al. Eur Heart J. 2018;39:1883-1948

- **AHA/ACC/HRS Syncope Statement (2017)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000000499
  - Shen WK, et al. Circulation. 2017;136:e60-e122

**San Francisco Syncope Rule:**
- Validation: https://pubmed.ncbi.nlm.nih.gov/14747812/
- PMID: 14747812
- Quinn J, et al. Ann Emerg Med. 2004

### 15. Aortic Valve Disorders (Aortic Stenosis)

**Primary Guidelines:**
- **AHA/ACC Valvular Heart Disease Guideline (2020)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/CIR.0000000000000923
  - Otto CM, et al. Circulation. 2021;143:e72-e227

- **ESC/EACTS Valvular Heart Disease Guidelines (2021)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/43/7/561/42728482/ehab395.pdf
  - Vahanian A, et al. Eur Heart J. 2022;43:561-632

**TAVR vs SAVR:**
- PARTNER 3 Trial: https://pubmed.ncbi.nlm.nih.gov/30883058/
- Evolut Low Risk Trial: https://pubmed.ncbi.nlm.nih.gov/30883053/

### 20. Pulmonary Embolism

**Primary Guidelines:**
- **ESC Pulmonary Embolism Guidelines (2019)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/41/4/543/32797089/ehz405.pdf
  - Konstantinides SV, et al. Eur Heart J. 2020;41:543-603

- **CHEST Guideline for Antithrombotic Therapy for VTE (2021)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7938271/pdf/main.pdf
  - Stevens SM, et al. Chest. 2021;160:e545-e608

**Risk Scores:**
- **Wells Score for PE**
  - Validation: https://pubmed.ncbi.nlm.nih.gov/9818866/
  - PMID: 9818866
  - Wells PS, et al. Thromb Haemost. 2000

- **PESI and sPESI**
  - Original PESI: https://pubmed.ncbi.nlm.nih.gov/15673557/
  - Simplified: https://pubmed.ncbi.nlm.nih.gov/21030640/

---

## INFECTIOUS DISEASE DIAGNOSES

### 7. Sepsis

**Primary Guidelines:**
- **Surviving Sepsis Campaign Guidelines (2021)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8623557/pdf/ccm-49-e1063.pdf
  - Evans L, et al. Crit Care Med. 2021;49:e1063-e1143
  - **CRITICAL** - Hour-1 bundle, fluid resuscitation, vasopressors

**Sepsis-3 Definition:**
- **Third International Consensus Definitions (2016)**
  - Direct PDF: https://jamanetwork.com/journals/jama/articlepdf/2492881/jama_singer_2016_sc_160035.pdf
  - Singer M, et al. JAMA. 2016;315:801-810
  - **Defines SOFA, qSOFA, septic shock**

**SOFA Score:**
- Original publication: https://pubmed.ncbi.nlm.nih.gov/8841913/
- Vincent JL, et al. Intensive Care Med. 1996

### 8. Pneumonia

**Community-Acquired Pneumonia (CAP):**
- **IDSA/ATS CAP Guidelines (2019)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6913800/pdf/ciz1092.pdf
  - Metlay JP, et al. Clin Infect Dis. 2019;69:e1-e33
  - **Includes CURB-65, PSI, treatment algorithms**

**Hospital-Acquired/Ventilator-Associated Pneumonia:**
- **IDSA/ATS HAP/VAP Guidelines (2016)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5840012/pdf/civ287.pdf
  - Kalil AC, et al. Clin Infect Dis. 2016;63:e61-e111

**CURB-65:**
- Original paper: https://pubmed.ncbi.nlm.nih.gov/14624108/
- Lim WS, et al. Thorax. 2003

### 9. Urinary Tract Infection (UTI)

**Primary Guidelines:**
- **IDSA Guidelines for Uncomplicated Cystitis and Pyelonephritis (2011, reaffirmed 2019)**
  - Direct PDF: https://academic.oup.com/cid/article-pdf/52/5/e103/441348/cir096.pdf
  - Gupta K, et al. Clin Infect Dis. 2011;52:e103-e120

- **IDSA Catheter-Associated UTI Guidelines (2009)**
  - Direct PDF: https://academic.oup.com/cid/article-pdf/50/5/625/441502/ciq043.pdf
  - Hooton TM, et al. Clin Infect Dis. 2010;50:625-663

**Asymptomatic Bacteriuria:**
- IDSA Guideline: https://pubmed.ncbi.nlm.nih.gov/30895288/
- Nicolle LE, et al. Clin Infect Dis. 2019

---

## RENAL DIAGNOSIS

### 10. Acute Kidney Injury (AKI)

**Primary Guidelines:**
- **KDIGO AKI Clinical Practice Guideline (2012)**
  - Direct PDF: https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf
  - Kidney Int Suppl. 2012;2:1-138
  - **Defines KDIGO staging, management**

- **KDIGO Hepatorenal Syndrome Update (2021)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8263840/pdf/kir20211.pdf
  - Kidney Int. 2021;100:S1-S184

**FENa and Renal Failure Indices:**
- Classic paper: https://pubmed.ncbi.nlm.nih.gov/7053744/
- Miller TR, et al. Ann Intern Med. 1978

---

## PSYCHIATRIC DIAGNOSES

### 11. Depression (Major Depressive Disorder)

**Primary Guidelines:**
- **APA Practice Guideline for Major Depressive Disorder (2010)**
  - Direct PDF: https://psychiatryonline.org/pb/assets/raw/sitewide/practice_guidelines/guidelines/mdd.pdf
  - Am J Psychiatry. 2010;167(suppl)

**DSM-5 Criteria:**
- American Psychiatric Association. Diagnostic and Statistical Manual of Mental Disorders, 5th ed. 2013

**PHQ-9:**
- Validation: https://pubmed.ncbi.nlm.nih.gov/11556941/
- PMID: 11556941
- Kroenke K, et al. J Gen Intern Med. 2001

**Treatment Algorithms:**
- STAR*D Trial: https://pubmed.ncbi.nlm.nih.gov/16717171/
- Trivedi MH, et al. Am J Psychiatry. 2006

### 12. Alcohol Use Disorder

**Primary Guidelines:**
- **ASAM National Practice Guideline (2020)**
  - Available at: https://www.asam.org/quality-care/clinical-guidelines/alcohol
  - Comprehensive treatment guideline

**DSM-5 Criteria:**
- American Psychiatric Association. DSM-5. 2013

**Withdrawal Management:**
- **ASAM Clinical Practice Guideline on Alcohol Withdrawal (2020)**
  - J Addict Med. 2020;14(3S Suppl 1):1-72

**CAGE Questionnaire:**
- Original: https://pubmed.ncbi.nlm.nih.gov/4031900/
- Ewing JA. JAMA. 1984

**CIWA-Ar:**
- Validation: https://pubmed.ncbi.nlm.nih.gov/2597811/
- Sullivan JT, et al. Br J Addict. 1989

### 17. Psychosis

**Schizophrenia Guidelines:**
- **APA Practice Guideline for Schizophrenia (2020)**
  - Direct PDF: https://psychiatryonline.org/pb/assets/raw/sitewide/practice_guidelines/guidelines/schizophrenia.pdf
  - Am J Psychiatry. 2020;177:868-872

**First-Episode Psychosis:**
- RAISE-ETP Trial: https://pubmed.ncbi.nlm.nih.gov/25219520/
- Kane JM, et al. Am J Psychiatry. 2015

**Clozapine Guidelines:**
- Monitoring protocol: https://pubmed.ncbi.nlm.nih.gov/31337255/
- Nielsen J, et al. CNS Drugs. 2018

---

## NEUROLOGICAL DIAGNOSIS

### 18. Stroke (Acute Ischemic Stroke)

**Primary Guidelines:**
- **AHA/ASA Acute Ischemic Stroke Guidelines (2019)**
  - Direct PDF: https://www.ahajournals.org/doi/pdf/10.1161/STR.0000000000000211
  - Powers WJ, et al. Stroke. 2019;50:e344-e418

- **ESC/ESO Stroke Guidelines (2021)**
  - Direct PDF: https://academic.oup.com/eurheartj/article-pdf/42/38/3576/40697353/ehab309.pdf
  - Berge E, et al. Eur Heart J. 2021;42:3576-3583

**tPA and Thrombectomy:**
- NINDS tPA Trial: https://pubmed.ncbi.nlm.nih.gov/7854284/
- DAWN Trial: https://pubmed.ncbi.nlm.nih.gov/29129157/
- DEFUSE 3: https://pubmed.ncbi.nlm.nih.gov/29129158/

**NIHSS:**
- Original scale: https://pubmed.ncbi.nlm.nih.gov/2549549/
- Brott T, et al. Stroke. 1989

---

## RESPIRATORY DIAGNOSIS

### 19. COPD

**Primary Guidelines:**
- **GOLD COPD Report (2024)**
  - Direct PDF: https://goldcopd.org/2024-gold-report/
  - Global Initiative for Chronic Obstructive Lung Disease
  - **CRITICAL** - ABCD assessment, treatment algorithms

- **ATS/ERS COPD Statement**
  - Review: https://pubmed.ncbi.nlm.nih.gov/15256389/
  - Celli BR, et al. Eur Respir J. 2004

**COPD Exacerbation:**
- ERS/ATS Guidelines: https://pubmed.ncbi.nlm.nih.gov/28128191/
- Wedzicha JA, et al. Am J Respir Crit Care Med. 2017

**mMRC Dyspnea Scale:**
- Validation: https://pubmed.ncbi.nlm.nih.gov/10193912/
- Mahler DA, Wells CK. Am Rev Respir Dis. 1988

---

## GASTROINTESTINAL DIAGNOSIS

### 16. Acute Pancreatitis

**Primary Guidelines:**
- **ACG Clinical Guideline on Acute Pancreatitis (2013)**
  - Direct PDF: https://journals.lww.com/ajg/fulltext/2013/09000/acg_clinical_guideline__management_of_acute.25.aspx
  - Tenner S, et al. Am J Gastroenterol. 2013;108:1400-1415

- **IAP/APA Evidence-Based Guidelines (2013)**
  - Direct PDF: https://link.springer.com/content/pdf/10.1007/s00595-015-1109-5.pdf
  - Working Group IAP/APA. Pancreatology. 2013

**Revised Atlanta Classification (2012):**
- Definition: https://pubmed.ncbi.nlm.nih.gov/23079836/
- Banks PA, et al. Gut. 2013

**Ranson Criteria:**
- Original: https://pubmed.ncbi.nlm.nih.gov/4814003/
- Ranson JH, et al. Surg Gynecol Obstet. 1974

---

## ONCOLOGY DIAGNOSIS

### 13. Chemotherapy Encounter

**NCCN Guidelines (Free registration required):**
- Available at: https://www.nccn.org/professionals/physician_gls/
- Cancer-specific treatment guidelines

**Supportive Care:**
- **ASCO/ONS Antiemetic Guideline (2020)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7526506/pdf/JCO.20.01296.pdf
  - Hesketh PJ, et al. J Clin Oncol. 2020;38:2782-2797

- **ASCO/IDSA Febrile Neutropenia Guideline (2018)**
  - Direct PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6804020/pdf/JCO.18.00269.pdf
  - Taplitz RA, et al. J Clin Oncol. 2018;36:3043-3054

**Tumor Lysis Syndrome:**
- Management: https://pubmed.ncbi.nlm.nih.gov/18658229/
- Cairo MS, et al. Br J Haematol. 2010

**Chemotherapy-Induced Cardiotoxicity:**
- ESC Cardio-Oncology: https://academic.oup.com/eurheartj/article/43/41/4229/6670402
- Lyon AR, et al. Eur Heart J. 2022

---

## ADDITIONAL RESOURCES

### Medical Calculators
- **MDCalc**: https://www.mdcalc.com/
  - All major clinical scores (HEART, TIMI, SOFA, CURB-65, CHA2DS2-VASc, etc.)
  - Evidence-based, peer-reviewed

- **QxMD Calculate**: https://qxmd.com/calculate
  - Mobile-friendly calculators

### Systematic Reviews and Meta-Analyses
- **Cochrane Library**: https://www.cochranelibrary.com/
  - High-quality systematic reviews

### Clinical Trial Databases
- **ClinicalTrials.gov**: https://clinicaltrials.gov/
  - All registered clinical trials

### Drug Information
- **Epocrates**: https://www.epocrates.com/
- **UpToDate**: https://www.uptodate.com/ (subscription)

### Textbooks (Open Access)
- **StatPearls**: https://www.ncbi.nlm.nih.gov/books/NBK430685/
  - Comprehensive, peer-reviewed, frequently updated
  - Covers all 20 diagnoses

---

## Implementation Notes

### For ClinOrchestra RAG Tab:

1. **Download Priority PDFs**:
   - Sepsis: Surviving Sepsis Campaign 2021
   - MI: Fourth Universal Definition
   - Heart Failure: AHA/ACC 2022
   - Stroke: AHA/ASA 2019
   - COPD: GOLD 2024
   - Pneumonia: IDSA/ATS CAP 2019
   - AKI: KDIGO 2012

2. **Upload to RAG**:
   - Use "Add Document from File" for downloaded PDFs
   - Use "Add Document from URL" for StatPearls chapters
   - Set k_value=3 (retrieve top 3 chunks)
   - Set chunk_size=512 (optimal for clinical guidelines)

3. **Organize by Category**:
   - Create separate RAG configurations for each category
   - Or combine all into single RAG index (recommended for cross-diagnosis learning)

### Citation Format
When using these resources in publications, cite:
- Original guideline authors
- ClinOrchestra system
- MIMIC-IV database

---

**Last Updated**: 2025-11-12
**Maintainer**: Frederick Gyasi (gyasi@musc.edu)
**Institution**: Medical University of South Carolina, Biomedical Informatics Center

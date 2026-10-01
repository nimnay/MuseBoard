# Product Requirements Document: CuratorAgent

Status: Draft  
Target Release: MVP (2 days)  
Author: [Your Name]  

---

## 1. Overview & Problem Statement

Content creators, marketers, and designers frequently use Pinterest as a visual discovery tool. However, Pinterest remains an unstructured collection of images, requiring manual effort to convert inspiration into actionable marketing systems such as:

- Content calendars  
- Brand identity guidelines  
- Color palettes  
- Structured Notion databases  

This process is time-consuming, subjective, and non-scalable.

**CuratorAgent** automates this workflow by transforming Pinterest boards into structured marketing intelligence using multimodal AI, embeddings, and automated Notion integration.

---

## 2. Goals & Success Metrics

### Goals

- Build an end-to-end autonomous pipeline from image ingestion → structured output.
- Demonstrate multimodal AI usage (vision + embeddings + LLM reasoning).
- Showcase real-world API orchestration (Pinterest + Notion + AI APIs).

---

### Success Metrics (MVP)

- Successfully ingest up to 50 pins from a Pinterest board.
- Generate structured metadata with ≥90% schema validity rate from vision model outputs.
- Automatically populate a Notion database with extracted insights.
- Execute full pipeline via CLI with zero manual intervention.

---

## 3. User Personas

### Freelance Marketer
Transforms client Pinterest mood boards into structured brand strategy documents and content calendars.

### Content Creator
Analyzes visual trends from inspiration boards to generate monthly posting strategies for Instagram, TikTok, or blogs.

---

## 4. User Stories

| Persona | Action | Expected Outcome |
|--------|--------|------------------|
| Marketer | Inputs Pinterest board URL | System ingests all pins reliably |
| Creator | Requests theme extraction | System identifies colors, objects, and aesthetic patterns |
| Marketer | Reviews Notion export | Structured database with clusters, tags, and strategy text |

---

## 5. System Architecture

Pinterest Board  
→ Ingestion Layer  
→ Vision Model (image understanding)  
→ Structured JSON Extraction  
→ Embedding Model  
→ Vector Store + Clustering  
→ LLM Strategy Synthesis  
→ Notion Database Export  

---

## 6. Functional Requirements (MVP)

### Phase 1: Ingestion Pipeline

- Authenticate with Pinterest API v5 (trial access)
- Fetch up to 50 pins per board
- Normalize raw API responses into internal schema
- Implement local caching layer (JSON or SQLite) for development stability
- Handle API rate limits with exponential backoff

---

### Phase 2: Vision & Feature Extraction

Each image is processed using a multimodal LLM (GPT-4o or Claude 3.5 Sonnet).

#### Required Output Schema (Strict)

Each pin must return:

```json
{
  "primary_objects": [],
  "color_palette": [],
  "aesthetic_tags": [],
  "mood": "",
  "style_category": "",
  "composition_notes": "",
  "text_detected": ""
}
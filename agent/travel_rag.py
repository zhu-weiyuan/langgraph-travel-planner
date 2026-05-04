# -*- coding: utf-8 -*-
"""
Travel RAG Module — BM25-based retrieval for travel knowledge base.

Loads markdown travel guides from knowledge/ directory and provides
semantic search over attractions, restaurants, tips, and practical info.
"""

import math
import re
from pathlib import Path
from typing import List, Dict

# Knowledge base directory
KB_DIR = Path(__file__).parent.parent / "knowledge"

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Cached documents and indices
_documents = []
_bm25_index = {}
_doc_lengths = []
_avg_doc_length = 0.0


def _tokenize(text: str) -> List[str]:
    """Simple Chinese tokenizer — bigrams + trigrams + single chars."""
    tokens = []
    # Extract Chinese runs
    chinese_runs = re.findall(r'[\u4e00-\u9fff]+', text)
    for run in chinese_runs:
        if len(run) >= 3:
            for i in range(len(run) - 2):
                tokens.append(run[i:i+3])
        if len(run) >= 2:
            for i in range(len(run) - 1):
                tokens.append(run[i:i+2])
        for c in run:
            tokens.append(c)
    # English words
    tokens.extend(re.findall(r'[a-zA-Z]+', text))
    return [t.lower() for t in tokens if t.strip()]


def _load_knowledge_base() -> List[dict]:
    """Load all markdown files from knowledge/ directory."""
    global _documents, _bm25_index, _doc_lengths, _avg_doc_length

    if _documents:
        return _documents

    _documents = []
    if not KB_DIR.exists():
        print(f"[Travel RAG] Knowledge base not found: {KB_DIR}")
        return _documents

    for md_file in sorted(KB_DIR.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
            doc = _parse_markdown(text, md_file.stem)
            _documents.append(doc)
            print(f"[Travel RAG] Loaded: {md_file.name} ({len(doc['sections'])} sections)")
        except Exception as e:
            print(f"[Travel RAG] Error loading {md_file}: {e}")

    _build_index()
    return _documents


def _parse_markdown(text: str, title: str) -> dict:
    """Parse markdown into sections based on headings."""
    sections = []
    lines = text.split("\n")
    current_heading = title
    current_content = []

    for line in lines:
        if line.startswith("#"):
            if current_content:
                sections.append({
                    "title": current_heading,
                    "text": "\n".join(current_content).strip(),
                })
            heading_text = line.lstrip("# ").strip()
            if heading_text:
                current_heading = heading_text
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "title": current_heading,
            "text": "\n".join(current_content).strip(),
        })

    return {"title": title, "sections": sections}


def _build_index():
    """Build BM25 inverted index."""
    global _bm25_index, _doc_lengths, _avg_doc_length

    _bm25_index = {}
    _doc_lengths = []

    section_id = 0
    for doc in _documents:
        for section in doc["sections"]:
            # Title weighted 3x
            title_tokens = _tokenize(section["title"]) * 3
            text_tokens = _tokenize(section["text"])
            combined = title_tokens + text_tokens

            tf = {}
            for token in combined:
                tf[token] = tf.get(token, 0) + 1

            _doc_lengths.append(len(combined))

            for word, count in tf.items():
                if word not in _bm25_index:
                    _bm25_index[word] = {}
                _bm25_index[word][section_id] = count

            section_id += 1

    if _doc_lengths:
        _avg_doc_length = sum(_doc_lengths) / len(_doc_lengths)

    print(f"[Travel RAG] Index built: {len(_bm25_index)} terms, {section_id} sections")


def _bm25_score(query_tokens: List[str], doc_id: int) -> float:
    """Compute BM25 score for a document."""
    N = len(_doc_lengths)
    if N == 0 or doc_id >= N:
        return 0.0

    doc_len = _doc_lengths[doc_id]
    score = 0.0

    for q_token in query_tokens:
        if q_token not in _bm25_index:
            continue

        n_q = len(_bm25_index[q_token])
        if n_q == 0:
            continue

        idf = math.log((N - n_q + 0.5) / (n_q + 0.5))
        if idf < 0:
            idf = 0.01

        tf = _bm25_index[q_token].get(doc_id, 0)
        if tf == 0:
            continue

        numerator = tf * (BM25_K1 + 1)
        denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(_avg_doc_length, 1))

        score += idf * (numerator / denominator)

    return score


def _get_section(doc_id: int) -> dict:
    """Get section by global ID."""
    count = 0
    for doc in _documents:
        for section in doc["sections"]:
            if count == doc_id:
                return section
            count += 1
    return None


def retrieve(query: str, top_k: int = 3) -> List[dict]:
    """Retrieve most relevant travel guide sections.

    Args:
        query: Search query (city name, attraction, food, etc.)
        top_k: Number of results to return

    Returns:
        List of {"title", "text", "score", "source"} dicts.
    """
    docs = _load_knowledge_base()
    if not docs:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Score each section
    scored_sections = []
    section_id = 0
    for doc in docs:
        for section in doc["sections"]:
            score = _bm25_score(query_tokens, section_id)
            if score > 0:
                scored_sections.append({
                    "title": section["title"],
                    "text": section["text"],
                    "score": round(score, 4),
                    "source": doc["title"],
                })
            section_id += 1

    # Sort by score descending
    scored_sections.sort(key=lambda x: x["score"], reverse=True)

    # Source diversity: max 2 per source
    diverse_results = []
    source_counts = {}
    for result in scored_sections:
        source = result["source"]
        if source_counts.get(source, 0) < 2:
            diverse_results.append(result)
            source_counts[source] = source_counts.get(source, 0) + 1
        if len(diverse_results) >= top_k:
            break

    return diverse_results


def build_context(query: str, max_length: int = 1500) -> str:
    """Build RAG context string for travel queries.

    Args:
        query: Search query
        max_length: Maximum context length

    Returns:
        Formatted context string with relevant travel info.
    """
    results = retrieve(query, top_k=3)
    if not results:
        return ""

    parts = ["\n## 参考资料（旅游攻略知识库）\n"]
    total_length = len(parts[0])

    for i, section in enumerate(results, 1):
        section_text = section["text"]
        if len(section_text) > 500:
            section_text = section_text[:500] + "..."

        block = f"\n### [{i}] {section['title']}\n{section_text}\n"

        if total_length + len(block) > max_length and i > 1:
            break

        parts.append(block)
        total_length += len(block)

    return "".join(parts)


def reload():
    """Force reload knowledge base."""
    global _documents, _bm25_index, _doc_lengths, _avg_doc_length
    _documents = []
    _bm25_index = {}
    _doc_lengths = []
    _avg_doc_length = 0.0
    return _load_knowledge_base()

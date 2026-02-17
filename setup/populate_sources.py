#!/usr/bin/env python3
"""Populate the data_sources table with 35 texts across 7 categories.

Fetches from:
- Wikipedia REST API for encyclopedia articles
- Python docs for coding documentation
- GitHub raw files for source code
- Various public sources for sports, social science, education
- Inline AI-generated text (clearly labeled)

Also writes each text to source_text/ for use with encode_story.py.
"""

import sqlite3
import os
import requests
import time
from datetime import datetime, timezone

DB_FILE = "benchmarks.db"
SOURCE_DIR = "source_text"
TIMESTAMP = datetime.now(timezone.utc).isoformat()

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_url(url, max_chars=4000):
    """Fetch text from a URL, truncating to max_chars."""
    resp = requests.get(url, timeout=30, headers={
        "User-Agent": "LLMzip-research/1.0 (academic compression benchmark)"
    })
    resp.raise_for_status()
    text = resp.text[:max_chars]
    return text.strip()


def fetch_wikipedia(title, max_chars=4000):
    """Fetch plain-text extract of a Wikipedia article via REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    resp = requests.get(url, timeout=30, headers={
        "User-Agent": "LLMzip-research/1.0 (academic compression benchmark)"
    })
    resp.raise_for_status()
    data = resp.json()
    extract = data.get("extract", "")

    # If summary is too short, try the full extract endpoint
    if len(extract) < 500:
        url2 = f"https://en.wikipedia.org/api/rest_v1/page/mobile-html/{title}"
        # Fall back to the TextExtracts API
        url2 = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title.replace("_", " "),
            "prop": "extracts",
            "explaintext": True,
            "exintro": False,
            "format": "json",
        }
        resp2 = requests.get(url2, params=params, timeout=30, headers={
            "User-Agent": "LLMzip-research/1.0 (academic compression benchmark)"
        })
        resp2.raise_for_status()
        pages = resp2.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", extract)

    return extract[:max_chars].strip()


def insert_source(cursor, category, title, source_url, text_content):
    """Insert a data source into the database."""
    cursor.execute('''
        INSERT INTO data_sources (category, title, source_url, text_content, collected_at, char_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (category, title, source_url, text_content, TIMESTAMP, len(text_content)))

    # Also write to source_text/ for benchmark use
    slug = f"{category}_{title.lower().replace(' ', '_').replace('/', '_')[:40]}"
    filepath = os.path.join(SOURCE_DIR, f"{slug}.txt")
    with open(filepath, 'w') as f:
        f.write(text_content)
    print(f"  [{category}] {title} ({len(text_content)} chars) -> {filepath}")


# ── Source Definitions ───────────────────────────────────────────────────────

WIKIPEDIA_ARTICLES = [
    ("Photosynthesis", "https://en.wikipedia.org/wiki/Photosynthesis"),
    ("General_relativity", "https://en.wikipedia.org/wiki/General_relativity"),
    ("French_Revolution", "https://en.wikipedia.org/wiki/French_Revolution"),
    ("DNA", "https://en.wikipedia.org/wiki/DNA"),
    ("Byzantine_Empire", "https://en.wikipedia.org/wiki/Byzantine_Empire"),
]

CODING_DOCS = [
    ("Python itertools", "https://docs.python.org/3/library/itertools.html",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/itertools.rst"),
    ("Python asyncio Tasks", "https://docs.python.org/3/library/asyncio-task.html",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/asyncio-task.rst"),
    ("Python collections", "https://docs.python.org/3/library/collections.html",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/collections.rst"),
    ("Python sqlite3", "https://docs.python.org/3/library/sqlite3.html",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/sqlite3.rst"),
    ("Python json", "https://docs.python.org/3/library/json.html",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/json.rst"),
]

SOURCE_CODE = [
    ("CPython json module", "https://raw.githubusercontent.com/python/cpython/main/Lib/json/__init__.py"),
    ("CPython pathlib module", "https://raw.githubusercontent.com/python/cpython/main/Lib/pathlib/_local.py"),
    ("Node.js fs module", "https://raw.githubusercontent.com/nodejs/node/main/lib/fs.js"),
    ("Flask app module", "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py"),
    ("Requests api module", "https://raw.githubusercontent.com/psf/requests/main/src/requests/api.py"),
]

SOCIAL_SCIENCE_SOURCES = [
    ("Dunbar's number and social networks",
     "https://en.wikipedia.org/api/rest_v1/page/summary/Dunbar%27s_number",
     "Dunbar's_number"),
    ("Tragedy of the commons",
     "https://en.wikipedia.org/api/rest_v1/page/summary/Tragedy_of_the_commons",
     "Tragedy_of_the_commons"),
    ("Prisoner's dilemma",
     "https://en.wikipedia.org/api/rest_v1/page/summary/Prisoner%27s_dilemma",
     "Prisoner's_dilemma"),
    ("Cognitive dissonance",
     "https://en.wikipedia.org/api/rest_v1/page/summary/Cognitive_dissonance",
     "Cognitive_dissonance"),
    ("Broken windows theory",
     "https://en.wikipedia.org/api/rest_v1/page/summary/Broken_windows_theory",
     "Broken_windows_theory"),
]

SPORTS_SOURCES = [
    ("History of the FIFA World Cup",
     "FIFA_World_Cup"),
    ("Olympic Games overview",
     "Olympic_Games"),
    ("History of basketball",
     "History_of_basketball"),
    ("Tour de France cycling",
     "Tour_de_France"),
    ("Cricket overview",
     "Cricket"),
]

EDUCATION_SOURCES = [
    ("Bloom's taxonomy",
     "Bloom%27s_taxonomy"),
    ("Montessori education",
     "Montessori_education"),
    ("History of education",
     "History_of_education"),
    ("Constructivism in education",
     "Constructivism_(philosophy_of_education)"),
    ("Socratic method",
     "Socratic_method"),
]

AI_GENERATED_TEXTS = [
    ("AI Essay: Climate Change Solutions",
     """Climate change represents one of the most pressing challenges facing humanity in the twenty-first century. The accumulation of greenhouse gases in the atmosphere, primarily carbon dioxide and methane, has led to a measurable increase in global average temperatures. Scientists have documented rising sea levels, more frequent extreme weather events, and shifts in precipitation patterns across every continent.

Addressing this crisis requires a multifaceted approach combining technological innovation, policy reform, and behavioral change. Renewable energy sources such as solar, wind, and geothermal power offer viable alternatives to fossil fuels. Solar panel efficiency has improved dramatically over the past decade, with modern photovoltaic cells converting over twenty-five percent of incoming sunlight into electricity. Wind turbines have similarly become more cost-effective, with offshore installations generating electricity at prices competitive with natural gas.

Carbon capture and storage technologies represent another promising avenue. These systems extract carbon dioxide directly from industrial emissions or ambient air, compressing it for underground storage in geological formations. While still expensive relative to emission reduction, ongoing research continues to drive down costs.

Policy mechanisms such as carbon pricing, cap-and-trade systems, and renewable energy mandates create economic incentives for businesses and individuals to reduce their carbon footprints. The European Union's Emissions Trading System, launched in 2005, has demonstrated that market-based approaches can achieve meaningful reductions when properly designed and enforced."""),

    ("AI Essay: The Future of Artificial Intelligence",
     """Artificial intelligence has undergone remarkable transformation since its conceptual origins in the mid-twentieth century. What began as theoretical explorations in mathematical logic and computation has evolved into a practical technology that touches nearly every aspect of modern life. From natural language processing to computer vision, from autonomous vehicles to medical diagnostics, AI systems now perform tasks that were once thought to require uniquely human intelligence.

The development of deep learning, powered by neural networks with millions or billions of parameters, has been particularly transformative. These models learn patterns from vast quantities of data, developing representations that capture subtle statistical regularities invisible to human analysts. Large language models, trained on extensive text corpora, can generate coherent prose, translate between languages, summarize documents, and answer questions across diverse domains.

However, significant challenges remain. Current AI systems lack genuine understanding of the concepts they manipulate. They can produce plausible-sounding text without comprehending its meaning, identify objects in images without understanding their physical properties, and optimize objectives without appreciating the broader consequences of their actions. This limitation, sometimes called the alignment problem, has prompted extensive research into making AI systems more reliable, interpretable, and aligned with human values.

The economic implications of artificial intelligence are equally profound. Automation threatens to displace workers in manufacturing, transportation, customer service, and even knowledge work. Simultaneously, AI creates new opportunities in data science, machine learning engineering, AI safety research, and countless application domains."""),

    ("AI Essay: Quantum Computing Fundamentals",
     """Quantum computing harnesses the principles of quantum mechanics to process information in fundamentally different ways than classical computers. While traditional computers store data as binary digits, or bits, that exist in one of two states, zero or one, quantum computers use quantum bits, or qubits, that can exist in superpositions of both states simultaneously.

This property of superposition allows quantum computers to explore multiple computational pathways in parallel. When combined with quantum entanglement, where the states of two or more qubits become correlated regardless of physical distance, quantum computers can solve certain problems exponentially faster than their classical counterparts.

The most well-known quantum algorithm is Shor's algorithm for integer factorization. Classical computers require exponentially increasing time to factor large numbers, a difficulty that underpins modern cryptographic systems like RSA. Shor's algorithm can factor these same numbers in polynomial time, potentially rendering current encryption methods obsolete once sufficiently powerful quantum computers become available.

Grover's algorithm provides a quadratic speedup for unstructured search problems. Given a database of N items, a classical computer requires on average N/2 queries to find a specific item, while Grover's algorithm accomplishes the same task in roughly the square root of N queries. This speedup has applications in optimization, machine learning, and database search.

Building practical quantum computers remains enormously challenging. Qubits are extremely sensitive to environmental disturbances, a phenomenon known as decoherence. Maintaining quantum states long enough to perform useful computations requires sophisticated error correction techniques and operating temperatures near absolute zero."""),

    ("AI Story: The Last Library",
     """The library stood at the edge of the old quarter, its stone walls dark with centuries of rain. Margaret pushed through the heavy oak door, breathing in that familiar scent of aged paper and wood polish that no digital archive could ever replicate.

She had been the head librarian for thirty-two years, arriving each morning at seven o'clock to sort the overnight returns and check the humidity sensors in the rare books room. The routine had become as natural as breathing, each day following the gentle rhythm of books shelved and books found, questions asked and questions answered.

But today was different. The letter from the city council lay folded in her cardigan pocket, its bureaucratic language barely concealing the finality of its message. The library would close at the end of the month. Budget constraints, declining visitors, the inevitable march of digital transformation. The words blurred together into a single verdict.

She walked through the stacks, trailing her fingers along the spines. Each shelf held its own geography of knowledge. The history section, where old Professor Hawkins had spent every Tuesday afternoon for two decades. The children's corner, where generations of young readers had discovered worlds beyond their neighborhoods. The reference desk, where she had helped thousands of patrons navigate the labyrinth of human knowledge.

The afternoon light slanted through the tall windows, casting long rectangles of gold across the reading tables. A few regulars sat in their customary spots. Arthur with his crossword puzzles. Mrs. Chen with her stack of gardening magazines. Young Theo, who came after school to do homework because his apartment was too noisy."""),

    ("AI Technical Doc: RESTful API Design",
     """RESTful API design follows a set of architectural principles that promote scalability, simplicity, and interoperability between distributed systems. REST, which stands for Representational State Transfer, was introduced by Roy Fielding in his doctoral dissertation in 2000 and has since become the dominant paradigm for web service development.

The fundamental concept in REST is the resource. Every piece of data or functionality exposed by an API is modeled as a resource, identified by a unique URI. Resources are manipulated through a standard set of HTTP methods: GET retrieves a resource's current state, POST creates a new resource, PUT replaces an existing resource entirely, PATCH applies partial modifications, and DELETE removes a resource.

Effective URI design follows consistent naming conventions. Resources should be represented as nouns rather than verbs. Collection endpoints use plural forms, such as /users or /articles, while individual resources are accessed by appending an identifier, such as /users/42. Nested resources express relationships, so /users/42/orders retrieves orders belonging to user 42.

HTTP status codes communicate the outcome of each request. Successful operations return codes in the 200 range: 200 for successful retrieval, 201 for successful creation, and 204 for successful deletion with no content returned. Client errors use the 400 range: 400 for malformed requests, 401 for authentication failures, 403 for authorization failures, and 404 for missing resources. Server errors use the 500 range.

Pagination, filtering, and sorting are essential for endpoints that return collections. Query parameters provide a clean mechanism: /users?page=2&limit=25&sort=name&order=asc. The response should include metadata about total count, current page, and links to adjacent pages, following patterns like HATEOAS or JSON API."""),
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SOURCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check if already populated
    cursor.execute("SELECT COUNT(*) FROM data_sources")
    existing = cursor.fetchone()[0]
    if existing > 0:
        print(f"data_sources already has {existing} rows. Skipping.")
        conn.close()
        return

    print(f"Populating data_sources at {TIMESTAMP}\n")
    errors = []

    # 1. Wikipedia articles
    print("=== Wikipedia Articles ===")
    for title, url in WIKIPEDIA_ARTICLES:
        try:
            text = fetch_wikipedia(title)
            insert_source(cursor, "wikipedia", title.replace("_", " "), url, text)
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"Wikipedia {title}: {e}")
            print(f"  ERROR: {title}: {e}")

    # 2. Coding documentation (from RST source on GitHub)
    print("\n=== Coding Documentation ===")
    for title, doc_url, raw_url in CODING_DOCS:
        try:
            text = fetch_url(raw_url)
            insert_source(cursor, "coding_docs", title, doc_url, text)
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"Coding docs {title}: {e}")
            print(f"  ERROR: {title}: {e}")

    # 3. Source code
    print("\n=== Source Code ===")
    for title, url in SOURCE_CODE:
        try:
            text = fetch_url(url, max_chars=5000)
            insert_source(cursor, "source_code", title, url, text)
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"Source code {title}: {e}")
            print(f"  ERROR: {title}: {e}")

    # 4. Social science (via Wikipedia detailed extracts)
    print("\n=== Social Science ===")
    for title, url, wiki_title in SOCIAL_SCIENCE_SOURCES:
        try:
            text = fetch_wikipedia(wiki_title)
            insert_source(cursor, "social_science", title, f"https://en.wikipedia.org/wiki/{wiki_title}", text)
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"Social science {title}: {e}")
            print(f"  ERROR: {title}: {e}")

    # 5. Sports information (via Wikipedia detailed extracts)
    print("\n=== Sports Information ===")
    for title, wiki_title in SPORTS_SOURCES:
        try:
            text = fetch_wikipedia(wiki_title)
            insert_source(cursor, "sports", title, f"https://en.wikipedia.org/wiki/{wiki_title}", text)
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"Sports {title}: {e}")
            print(f"  ERROR: {title}: {e}")

    # 6. Education content (via Wikipedia detailed extracts)
    print("\n=== Education ===")
    for title, wiki_title in EDUCATION_SOURCES:
        try:
            text = fetch_wikipedia(wiki_title)
            insert_source(cursor, "education", title, f"https://en.wikipedia.org/wiki/{wiki_title}", text)
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"Education {title}: {e}")
            print(f"  ERROR: {title}: {e}")

    # 7. AI-generated text
    print("\n=== AI-Generated Text ===")
    for title, text in AI_GENERATED_TEXTS:
        insert_source(cursor, "ai_generated", title, "ai_generated_by_llm", text)

    conn.commit()

    # Summary
    cursor.execute("SELECT category, COUNT(*), SUM(char_count) FROM data_sources GROUP BY category ORDER BY category")
    rows = cursor.fetchall()
    print(f"\n{'='*50}")
    print(f"{'Category':<20} {'Count':>5} {'Total Chars':>12}")
    print(f"{'-'*50}")
    for cat, count, chars in rows:
        print(f"{cat:<20} {count:>5} {chars:>12,}")
    total = sum(r[1] for r in rows)
    total_chars = sum(r[2] for r in rows)
    print(f"{'-'*50}")
    print(f"{'TOTAL':<20} {total:>5} {total_chars:>12,}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()

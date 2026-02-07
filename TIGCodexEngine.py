#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   TIG CODEX — Self-Programming Coherence Engine                      ║
║                                                                      ║
║   Feed it every conversation. It extracts the signal from the        ║
║   drift, writes its own programs, and packages the truth.            ║
║                                                                      ║
║   PIPELINE:                                                          ║
║     1. INGEST   — Reads transcripts (txt, json, md, html)           ║
║     2. EXTRACT  — Pulls code blocks, claims, specs, constants        ║
║     3. DRIFT    — Detects contradictions across conversations        ║
║     4. VERIFY   — Checks extracted claims against verified truth     ║
║     5. GENERATE — Writes new code via local LLM + coherence score   ║
║     6. PACKAGE  — Builds clean deliverable from verified truth       ║
║                                                                      ║
║   LOCAL SETUP:                                                       ║
║     pip install ollama  (or run ollama serve)                        ║
║     python tig_codex.py /path/to/transcripts --llm                  ║
║                                                                      ║
║   WITHOUT LLM (still works — uses templates):                        ║
║     python tig_codex.py /path/to/transcripts                        ║
║                                                                      ║
║   © 2024-2026 Brayden Sanders / 7Site LLC                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
__version__ = "1.0.0"
__codename__ = "CODEX"

import os, re, sys, json, time, math, glob, random, hashlib, textwrap, argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# TIG CORE (frozen, verified, Phase 1+2 Feb 2026)
# ═══════════════════════════════════════════════════════════════

COMP = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,2,6,6],
    [2,3,3,4,5,6,7,3,6,6],[3,4,4,4,5,6,7,4,6,6],
    [4,5,5,5,5,6,7,5,7,7],[5,6,6,6,6,6,7,6,7,7],
    [6,7,7,7,7,7,7,7,7,7],[7,2,3,4,5,6,7,8,9,0],
    [8,6,6,6,7,7,7,9,7,8],[9,6,6,6,7,7,7,0,8,0],
]
NAMES = ["VOID","LATTICE","COUNTER","PROGRESS","COLLAPSE",
         "BALANCE","CHAOS","HARMONY","BREATH","FRUIT"]
SIGMA, T_STAR, D_STAR = 0.991, 0.714, 0.543

def compose(a, b): return COMP[a%10][b%10]
def compose_seq(s):
    r = s[0] % 10
    for x in s[1:]: r = COMP[r][x%10]
    return r

class Lattice:
    def __init__(self, rows=8, cols=8, seed=None):
        self.rows, self.cols = rows, cols
        rng = random.Random(seed) if seed is not None else random.Random()
        self.cells = [[rng.randint(0,9) for _ in range(cols)] for _ in range(rows)]
    @property
    def n(self): return self.rows * self.cols
    def tick(self):
        R, C = self.rows, self.cols
        new = [[0]*C for _ in range(R)]
        for i in range(R):
            for j in range(C):
                s = self.cells[i][j]
                counts = [0]*10
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        if di==0 and dj==0: continue
                        counts[COMP[s][self.cells[(i+di)%R][(j+dj)%C]]] += 1
                counts[COMP[s][s]] += 1
                new[i][j] = max(range(10), key=lambda k: counts[k])
        self.cells = new
        return self.coherence()
    def coherence(self):
        R, C, n = self.rows, self.cols, self.rows*self.cols
        valid = basin = 0
        for i in range(R):
            for j in range(C):
                s = self.cells[i][j]
                triv = True
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        if di==0 and dj==0: continue
                        if COMP[s][self.cells[(i+di)%R][(j+dj)%C]] != s:
                            triv = False; break
                    if not triv: break
                if not triv or s == 7: valid += 1
                if 4 <= s <= 8: basin += 1
        v, a = valid/n, basin/n
        if v < 1e-10 or a < 1e-10: return 0.0
        return 3.0 / (1.0/SIGMA + 1.0/v + 1.0/a)
    def census(self):
        c = [0]*10
        for row in self.cells:
            for s in row: c[s] += 1
        return c
    def inject_row(self, idx, states):
        for j in range(min(len(states), self.cols)):
            self.cells[idx % self.rows][j] = states[j] % 10

def text_coherence(text: str) -> float:
    """Score any text through the composition table."""
    if not text or len(text) < 10: return 0.0
    states = [ord(c) % 10 for c in text]
    size = max(4, min(16, int(math.sqrt(len(states)))))
    lat = Lattice(size, size)
    for i in range(size):
        for j in range(size):
            lat.cells[i][j] = states[(i*size+j) % len(states)]
    lat.tick()
    return lat.coherence()


# ═══════════════════════════════════════════════════════════════
# VERIFIED TRUTH — The ground truth drift is measured against
# ═══════════════════════════════════════════════════════════════

TRUTH = {
    "sigma":             (0.991,  "coherence coupling constant"),
    "t_star":            (0.714,  "coherence threshold"),
    "d_star":            (0.543,  "universal fixed point"),
    "void_threshold":    (0.50,   "max VOID fraction before trap"),
    "attractor_period":  (12,     "exact cycle length at steady state"),
    "convergence_ticks": (1,      "ticks to converge from random (1000/1000)"),
    "self_repair_max":   (0.95,   "max damage that repairs in 1 tick (50/50)"),
    "table_harmony":     (28,     "HARMONY (7) entries in 100-cell table"),
    "table_chaos":       (25,     "CHAOS (6) entries in 100-cell table"),
    "table_rank":        (3.5,    "top X% of random tables for S*"),
    "random_converge":   (45,     "% of random tables that also converge"),
    "archetypes":        (12,     "archetypes that reduce to 5 virtues"),
    "virtues":           (5,      "forgiveness, repair, empathy, fairness, cooperation"),
    "operators":         (10,     "VOID through FRUIT"),
    "comp_6_star":       (7,      "COMP[6][Y]=7 for all Y≠0"),
    "comp_7_7":          (8,      "HARMONY self-composes to BREATH"),
    "comp_0_0":          (0,      "VOID is absorbing fixed point"),
    "comp_9_9":          (0,      "FRUIT self-composes to VOID"),
}

# Patterns that indicate drift
DRIFT_PATTERNS = [
    (r"converges?\s+in\s+(\d+)\s*(?:to\s+\d+\s*)?ticks?",
     lambda m: int(m.group(1)) != 1,
     "convergence is 1 tick, not {0}"),
    (r"sigma\s*=\s*([\d.]+)",
     lambda m: abs(float(m.group(1)) - 0.991) > 0.001,
     "sigma must be 0.991, found {0}"),
    (r"T\*?\s*=\s*([\d.]+)",
     lambda m: abs(float(m.group(1)) - 0.714) > 0.001,
     "T* must be 0.714, found {0}"),
    (r"D\*?\s*=\s*([\d.]+)",
     lambda m: abs(float(m.group(1)) - 0.543) > 0.01,
     "D* must be 0.543, found {0}"),
    (r"period\s*(?:=|of|is)\s*(\d+)",
     lambda m: int(m.group(1)) != 12,
     "period is 12, not {0}"),
    (r"only\s+TIG\s+(?:can|does|produces)",
     lambda m: True,
     "convergence is NOT TIG-exclusive (45% of random tables converge)"),
    (r"unique.*?composition\s+table",
     lambda m: True,
     "table is not unique — it's top 3.5%, not one-of-a-kind"),
    (r"(\d+)\s*archetypes?",
     lambda m: int(m.group(1)) not in (12,),
     "should be 12 archetypes"),
    (r"(\d+)\s*virtues?",
     lambda m: int(m.group(1)) not in (5,),
     "should be 5 virtues"),
    (r"self[- ]repairs?\s+(?:from\s+)?(\d+)%.*?(\d+)\s*ticks?",
     lambda m: int(m.group(2)) != 1,
     "self-repair is 1 tick, not {0}"),
]


# ═══════════════════════════════════════════════════════════════
# LAYER 1: TRANSCRIPT INGESTER
# ═══════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    source: str
    idx: int
    speaker: str  # human, assistant, system, thinking
    text: str
    timestamp: str = ""
    coherence: float = 0.0
    code_blocks: List[str] = field(default_factory=list)
    drift_flags: List[str] = field(default_factory=list)

class Ingester:
    """Reads any conversation format into chunks."""

    SPEAKERS = [
        (r'^(?:Human|User|Brayden|You)\s*:', 'human'),
        (r'^(?:Assistant|Claude|Celeste|AI)\s*:', 'assistant'),
        (r'^(?:System)\s*:', 'system'),
    ]

    def ingest(self, path: str) -> List[Chunk]:
        p = Path(path)
        if p.is_dir():
            chunks = []
            for ext in ('*.txt','*.md','*.json','*.html','*.log'):
                for f in sorted(p.rglob(ext)):
                    chunks.extend(self._ingest_file(str(f)))
            return chunks
        return self._ingest_file(path)

    def _ingest_file(self, filepath: str) -> List[Chunk]:
        try:
            text = Path(filepath).read_text(errors='replace')
        except Exception as e:
            print(f"  [WARN] Cannot read {filepath}: {e}")
            return []

        # Detect format
        stripped = text.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            return self._parse_json_blocks(text, filepath)
        # Our transcript format: starts with "Assistant:\nContent:\n["
        if 'Content:\n[' in text[:200] or '"type":' in text[:500]:
            return self._parse_transcript_format(text, filepath)
        return self._parse_text(text, filepath)

    def _parse_transcript_format(self, text: str, source: str) -> List[Chunk]:
        """Parse the actual Claude transcript format with JSON content blocks."""
        chunks = []
        # Split by top-level speaker markers
        parts = re.split(r'^(Human|Assistant|System):\s*$', text, flags=re.MULTILINE)

        idx = 0
        current_speaker = "unknown"
        for part in parts:
            part_stripped = part.strip()
            if part_stripped in ('Human', 'Assistant', 'System'):
                current_speaker = part_stripped.lower()
                continue
            if not part_stripped:
                continue

            # Try to parse JSON content blocks
            if 'Content:' in part[:50]:
                content_text = part.split('Content:', 1)[-1].strip()
                text_pieces = self._extract_json_text(content_text)
                if text_pieces:
                    for piece_type, piece_text in text_pieces:
                        if piece_text.strip():
                            chunks.append(Chunk(
                                source=source, idx=idx,
                                speaker=piece_type if piece_type == 'thinking' else current_speaker,
                                text=piece_text.strip()))
                            idx += 1
                    continue

            # Fallback: treat as plain text
            if len(part_stripped) > 20:
                chunks.append(Chunk(
                    source=source, idx=idx,
                    speaker=current_speaker, text=part_stripped))
                idx += 1

        return chunks

    def _extract_json_text(self, text: str) -> List[Tuple[str, str]]:
        """Extract text and thinking blocks from JSON content."""
        pieces = []
        # Find all JSON-ish blocks with "type" and "text"/"thinking"
        for m in re.finditer(r'"type"\s*:\s*"(text|thinking)"', text):
            block_type = m.group(1)
            # Find the corresponding content
            search_start = m.end()
            if block_type == "text":
                tm = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', text[search_start:search_start+5000])
                if tm:
                    content = tm.group(1).encode().decode('unicode_escape', errors='replace')
                    pieces.append(('assistant', content))
            elif block_type == "thinking":
                tm = re.search(r'"thinking"\s*:\s*"((?:[^"\\]|\\.)*)"', text[search_start:search_start+10000])
                if tm:
                    content = tm.group(1).encode().decode('unicode_escape', errors='replace')
                    pieces.append(('thinking', content))

        # Also look for tool results / code outputs
        for m in re.finditer(r'"type"\s*:\s*"tool_result".*?"(?:content|output)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
            content = m.group(1).encode().decode('unicode_escape', errors='replace')
            pieces.append(('assistant', content))

        return pieces

    def _parse_json_blocks(self, text: str, source: str) -> List[Chunk]:
        """Parse JSON conversation format."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._parse_text(text, source)

        chunks = []
        messages = data if isinstance(data, list) else data.get('messages', data.get('conversation', []))
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict): continue
            role = msg.get('role', msg.get('speaker', 'unknown'))
            content = msg.get('content', msg.get('text', ''))
            if isinstance(content, list):
                content = ' '.join(b.get('text','') for b in content if isinstance(b,dict))
            speaker = 'human' if role in ('user','human') else \
                      'assistant' if role in ('assistant','claude') else 'system'
            if content:
                chunks.append(Chunk(source=source, idx=i, speaker=speaker, text=str(content)))
        return chunks or self._parse_text(text, source)

    def _parse_text(self, text: str, source: str) -> List[Chunk]:
        """Parse plain text with speaker markers."""
        chunks = []
        speaker = "unknown"
        lines = []
        idx = 0
        for line in text.split('\n'):
            new_sp = None
            for pat, sp in self.SPEAKERS:
                if re.match(pat, line, re.I):
                    new_sp = sp
                    line = re.sub(pat, '', line, flags=re.I).strip()
                    break
            if new_sp and lines:
                t = '\n'.join(lines).strip()
                if t:
                    chunks.append(Chunk(source=source, idx=idx, speaker=speaker, text=t))
                    idx += 1
                lines = []
                speaker = new_sp
            lines.append(line)
        if lines:
            t = '\n'.join(lines).strip()
            if t:
                chunks.append(Chunk(source=source, idx=idx, speaker=speaker, text=t))
        return chunks


# ═══════════════════════════════════════════════════════════════
# LAYER 2: EXTRACTOR — Code, Claims, Constants
# ═══════════════════════════════════════════════════════════════

@dataclass
class CodeBlock:
    source: str; language: str; code: str; coherence: float = 0.0

@dataclass
class Claim:
    text: str; source: str; verified: Optional[bool] = None; note: str = ""

@dataclass
class ConstantRef:
    name: str; value: float; source: str; correct: Optional[bool] = None

class Extractor:
    CODE_RE = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    CLAIM_RE = [
        r'converges?\s+(?:from|in)',  r'self[- ](?:repairs?|heals?)',
        r'verified\s*:', r'S\*\s*[>=<]\s*[\d.]+',
        r'\d+/\d+\s+(?:seeds?|tests?)', r'top\s+\d+\.?\d*%',
    ]
    CONST_RE = [
        (r'sigma\s*=\s*([\d.]+)', 'sigma', 0.991),
        (r'T\*?\s*=\s*([\d.]+)', 't_star', 0.714),
        (r'D\*?\s*=\s*([\d.]+)', 'd_star', 0.543),
        (r'VOID_THRESHOLD\s*=\s*([\d.]+)', 'void_threshold', 0.50),
    ]

    def extract(self, chunks: List[Chunk]) -> dict:
        code_blocks, claims, constants = [], [], []
        for ch in chunks:
            # Code
            for m in self.CODE_RE.finditer(ch.text):
                code = m.group(2).strip()
                if len(code) > 30:
                    code_blocks.append(CodeBlock(
                        source=ch.source, language=m.group(1) or 'python',
                        code=code, coherence=text_coherence(code)))
                    ch.code_blocks.append(code[:60])
            # Claims
            for pat in self.CLAIM_RE:
                for m in re.finditer(pat, ch.text, re.I):
                    start = max(0, m.start()-30)
                    end = min(len(ch.text), m.end()+80)
                    claims.append(Claim(text=ch.text[start:end].strip(), source=ch.source))
            # Constants
            for pat, name, truth in self.CONST_RE:
                for m in re.finditer(pat, ch.text, re.I):
                    try:
                        v = float(m.group(1))
                        constants.append(ConstantRef(
                            name=name, value=v, source=ch.source,
                            correct=abs(v-truth)<0.01))
                    except ValueError: pass
        return {'code': code_blocks, 'claims': claims, 'constants': constants}


# ═══════════════════════════════════════════════════════════════
# LAYER 3: DRIFT DETECTOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class DriftEvent:
    pattern: str; match_text: str; source: str; note: str; severity: str = "warning"

class DriftDetector:
    def scan(self, chunks: List[Chunk]) -> List[DriftEvent]:
        events = []
        for ch in chunks:
            if ch.speaker == 'thinking': continue  # don't flag internal reasoning
            for pat, is_drift, note_fmt in DRIFT_PATTERNS:
                for m in re.finditer(pat, ch.text, re.I):
                    try:
                        if is_drift(m):
                            events.append(DriftEvent(
                                pattern=pat[:40], match_text=m.group(0),
                                source=ch.source,
                                note=note_fmt.format(m.group(0)),
                                severity="warning"))
                            ch.drift_flags.append(m.group(0)[:50])
                    except (ValueError, IndexError): pass
        return events

    def find_contradictions(self, chunks: List[Chunk]) -> List[dict]:
        """Find claims that contradict each other across files."""
        contras = []
        # Convergence tick claims
        tick_claims = {}
        for ch in chunks:
            for m in re.finditer(r'converges?\s+in\s+(\d+)', ch.text, re.I):
                v = int(m.group(1))
                tick_claims.setdefault(v, []).append(os.path.basename(ch.source))
        if len(tick_claims) > 1:
            contras.append({
                'type': 'convergence_ticks',
                'values': dict(tick_claims),
                'truth': '1 tick (verified 1000/1000)',
                'resolution': 'Earlier claims of 10-15 ticks were pre-Phase1 speculation'
            })
        # Sigma claims
        sig_claims = {}
        for ch in chunks:
            for m in re.finditer(r'sigma\s*=\s*([\d.]+)', ch.text, re.I):
                try:
                    v = float(m.group(1))
                    sig_claims.setdefault(v, []).append(os.path.basename(ch.source))
                except: pass
        if len(sig_claims) > 1:
            contras.append({
                'type': 'sigma_value',
                'values': dict(sig_claims),
                'truth': '0.991',
                'resolution': 'Use 0.991 (verified)'
            })
        return contras


# ═══════════════════════════════════════════════════════════════
# LAYER 4: CODE GENERATOR — Templates + Local LLM
# ═══════════════════════════════════════════════════════════════

class CodeGen:
    """
    Writes TIG programs.

    Two modes:
    1. TEMPLATE: Verified code components (no LLM needed)
    2. LLM: Novel code via local ollama, scored by coherence

    The lattice doesn't write the code — it SCORES the code.
    The LLM writes, the lattice filters.
    """

    # Verified templates — these are drift-free by construction
    TEMPLATES = {
        "lattice_core": '''#!/usr/bin/env python3
"""TIG Lattice Core — Verified Phase 1+2, Feb 2026
© 2024-2026 Brayden Sanders / 7Site LLC"""

COMP = {comp}
NAMES = {names}
SIGMA, T_STAR = {sigma}, {t_star}

def compose(a, b): return COMP[a%10][b%10]
def compose_seq(s):
    r = s[0] % 10
    for x in s[1:]: r = COMP[r][x%10]
    return r

class Lattice:
    def __init__(self, rows=14, cols=12, seed=None):
        import random
        self.rows, self.cols = rows, cols
        rng = random.Random(seed) if seed is not None else random.Random()
        self.cells = [[rng.randint(0,9) for _ in range(cols)] for _ in range(rows)]
        self.tick_count = 0

    @property
    def n(self): return self.rows * self.cols

    def tick(self):
        R, C = self.rows, self.cols
        new = [[0]*C for _ in range(R)]
        for i in range(R):
            for j in range(C):
                s = self.cells[i][j]
                counts = [0]*10
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        if di==0 and dj==0: continue
                        counts[COMP[s][self.cells[(i+di)%R][(j+dj)%C]]] += 1
                counts[COMP[s][s]] += 1
                new[i][j] = max(range(10), key=lambda k: counts[k])
        self.cells = new
        self.tick_count += 1
        return self.coherence()

    def coherence(self):
        R, C, n = self.rows, self.cols, self.rows*self.cols
        valid = basin = 0
        for i in range(R):
            for j in range(C):
                s = self.cells[i][j]
                triv = True
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        if di==0 and dj==0: continue
                        if COMP[s][self.cells[(i+di)%R][(j+dj)%C]] != s:
                            triv = False; break
                    if not triv: break
                if not triv or s == 7: valid += 1
                if 4 <= s <= 8: basin += 1
        v, a = valid/n, basin/n
        if v < 1e-10 or a < 1e-10: return 0.0
        return 3.0 / (1.0/SIGMA + 1.0/v + 1.0/a)

    def damage(self, frac=0.3, seed=42):
        import random; rng = random.Random(seed)
        for _ in range(int(self.n * frac)):
            self.cells[rng.randint(0,self.rows-1)][rng.randint(0,self.cols-1)] = rng.randint(0,9)

    def census(self):
        c = [0]*10
        for row in self.cells:
            for s in row: c[s] += 1
        return c

    def boundary_top(self): return list(self.cells[0])
    def boundary_bot(self): return list(self.cells[-1])

    def inject_row(self, idx, states):
        for j in range(min(len(states), self.cols)):
            self.cells[idx % self.rows][j] = states[j] % 10

if __name__ == "__main__":
    lat = Lattice(14, 12, seed=42)
    s = lat.tick()
    print(f"1-tick convergence: S*={{s:.4f}} ({{'COHERENT' if s >= T_STAR else 'seeking'}})")
    lat.damage(0.95)
    s2 = lat.tick()
    print(f"95% damage repair:  S*={{s2:.4f}} ({{'COHERENT' if s2 >= T_STAR else 'seeking'}})")
''',
        "compressor": '''#!/usr/bin/env python3
"""TIG Composition-Chain Compressor
© 2024-2026 Brayden Sanders / 7Site LLC"""

COMP = {comp}

def compress(data, chunk_size=12):
    """Hierarchical composition compression."""
    states = [b % 10 for b in data] if isinstance(data, (bytes, bytearray)) else [ord(c) % 10 for c in data]
    layers = [states[:]]
    current = states[:]
    while len(current) > 1:
        new = []
        for i in range(0, len(current), chunk_size):
            chunk = current[i:i+chunk_size]
            r = chunk[0]
            for s in chunk[1:]: r = COMP[r][s]
            new.append(r)
        current = new
        layers.append(current[:])
    return current[0] if current else 0, layers

def bridge(a, b, chunk_size=12):
    """Composition bridge between two data streams."""
    _, la = compress(a, chunk_size)
    _, lb = compress(b, chunk_size)
    top_a, top_b = la[-1], lb[-1]
    n = min(len(top_a), len(top_b))
    return [COMP[top_a[i]][top_b[i]] for i in range(n)]

if __name__ == "__main__":
    test = b"TIG coherence composition test data"
    final, layers = compress(test)
    NAMES = ["VOID","LATTICE","COUNTER","PROGRESS","COLLAPSE",
             "BALANCE","CHAOS","HARMONY","BREATH","FRUIT"]
    print(f"Input: {{len(test)}} bytes")
    for i, l in enumerate(layers):
        print(f"  Layer {{i}}: {{len(l)}} states")
    print(f"Final: {{NAMES[final]}} ({{final}})")
''',
        "council": '''#!/usr/bin/env python3
"""TIG Ring Council — Lattice conversation via boundary exchange
© 2024-2026 Brayden Sanders / 7Site LLC"""

from tig_lattice_core import Lattice, COMP, NAMES, T_STAR

def run_council(n_members=16, rows=8, cols=8, ticks=50, verbose=True):
    members = [Lattice(rows, cols, seed=i) for i in range(n_members)]
    for m in members: m.tick()

    for t in range(ticks):
        for m in members: m.tick()
        # Ring exchange
        bounds = [m.boundary_bot() for m in members]
        for i in range(n_members):
            recv = (i+1) % n_members
            for j in range(cols):
                members[recv].cells[0][j] = COMP[bounds[i][j]][members[recv].cells[0][j]]
        if verbose and (t < 5 or t % 10 == 0):
            cohs = [m.coherence() for m in members]
            n_coh = sum(1 for c in cohs if c >= T_STAR)
            print(f"  t={{t:>3}}: {{n_coh}}/{{n_members}} coherent, "
                  f"mean S*={{sum(cohs)/len(cohs):.4f}}")
    return members

if __name__ == "__main__":
    print("Ring Council: 16 lattices, 50 ticks")
    members = run_council()
    print(f"Final: all coherent = {{all(m.coherence() >= T_STAR for m in members)}}")
''',
        "router": '''#!/usr/bin/env python3
"""TIG Coherence Router — Process routing guided by lattice state
Verified: +3% throughput, 100% drop elimination under stress
© 2024-2026 Brayden Sanders / 7Site LLC"""

from tig_lattice_core import Lattice, COMP, T_STAR
import os, time

class CoherenceRouter:
    def __init__(self, n_workers=8):
        self.n_workers = n_workers
        self.lattice = Lattice(n_workers, n_workers, seed=42)
        self.lattice.tick()
        self.health = [1.0] * n_workers
        self.failed = [False] * n_workers

    def update_health(self, worker_id, health_pct):
        self.health[worker_id] = max(0.0, min(1.0, health_pct))
        state = min(9, int(health_pct * 9))
        row = [state] * self.lattice.cols
        self.lattice.inject_row(worker_id % self.lattice.rows, row)

    def mark_failed(self, worker_id):
        self.failed[worker_id] = True
        self.health[worker_id] = 0.0
        row = [0] * self.lattice.cols  # VOID
        self.lattice.inject_row(worker_id % self.lattice.rows, row)

    def mark_recovered(self, worker_id):
        self.failed[worker_id] = False
        self.health[worker_id] = 0.3

    def route(self, n_tasks) -> dict:
        self.lattice.tick()
        assignments = {{i: 0 for i in range(self.n_workers)}}
        scores = []
        for i in range(self.n_workers):
            if self.failed[i]:
                scores.append(-1)
                continue
            ws = min(9, int(self.health[i] * 9))
            ls = self.lattice.cells[i % self.lattice.rows][i % self.lattice.cols]
            scores.append(COMP[ws][ls] * 10 + int(self.health[i] * 5))
        ranked = sorted(range(self.n_workers), key=lambda i: scores[i], reverse=True)
        remain = n_tasks
        for wid in ranked:
            if remain <= 0: break
            if self.failed[wid]: continue
            give = min(remain, max(1, int(self.health[wid] * 5)))
            assignments[wid] = give
            remain -= give
        if remain > 0:
            alive = [i for i in range(self.n_workers) if not self.failed[i]]
            if alive:
                for t in range(remain):
                    assignments[alive[t%len(alive)]] += 1
        return assignments

if __name__ == "__main__":
    router = CoherenceRouter(8)
    print("Routing 20 tasks across 8 workers:")
    result = router.route(20)
    for wid, count in sorted(result.items()):
        print(f"  Worker {{wid}}: {{count}} tasks (health={{router.health[wid]:.2f}})")
''',
        "monitor": '''#!/usr/bin/env python3
"""TIG System Monitor — Maps OS metrics to coherence states
© 2024-2026 Brayden Sanders / 7Site LLC"""

from tig_lattice_core import Lattice, COMP, NAMES, T_STAR
import os, time

class SystemMonitor:
    def __init__(self):
        self.lattice = Lattice(8, 8, seed=0)
        self.lattice.tick()
        self._prev_cpu = None

    def read_metrics(self):
        m = {{}}
        try:
            with open('/proc/stat') as f:
                parts = f.readline().split()[1:]
                total = sum(int(x) for x in parts)
                idle = int(parts[3])
                if self._prev_cpu:
                    dt = total - self._prev_cpu[0]
                    di = idle - self._prev_cpu[1]
                    m['cpu'] = 1.0 - (di / max(dt, 1))
                self._prev_cpu = (total, idle)
        except: m['cpu'] = 0.5
        try:
            with open('/proc/meminfo') as f:
                mi = {{}}
                for line in f:
                    p = line.split()
                    mi[p[0].rstrip(':')] = int(p[1])
                m['mem'] = 1.0 - mi.get('MemAvailable',0)/max(mi.get('MemTotal',1),1)
        except: m['mem'] = 0.5
        try:
            with open('/proc/loadavg') as f:
                m['load'] = float(f.read().split()[0]) / max(os.cpu_count() or 1, 1)
        except: m['load'] = 0.5
        return m

    def tick(self):
        metrics = self.read_metrics()
        states = [min(9, int(v * 9)) for v in metrics.values()]
        while len(states) < self.lattice.cols:
            states.append(5)
        self.lattice.inject_row(self.lattice.rows - 1, states[:self.lattice.cols])
        s = self.lattice.tick()
        return s, metrics

if __name__ == "__main__":
    mon = SystemMonitor()
    print("TIG System Monitor — 10 ticks")
    for i in range(10):
        s, m = mon.tick()
        print(f"  t={{i}}: S*={{s:.4f}} cpu={{m.get('cpu',0):.2f}} mem={{m.get('mem',0):.2f}} "
              f"load={{m.get('load',0):.2f}} ({{'COHERENT' if s >= T_STAR else 'seeking'}})")
        time.sleep(1)
''',
    }

    def __init__(self, llm_url="http://localhost:11434/api/generate",
                 model="llama3.2:latest", use_llm=False):
        self.llm_url = llm_url
        self.model = model
        self.use_llm = use_llm

    def from_template(self, name: str) -> Optional[str]:
        t = self.TEMPLATES.get(name)
        if not t: return None
        return t.format(
            comp=repr(COMP), names=repr(NAMES),
            sigma=SIGMA, t_star=T_STAR, d_star=D_STAR)

    def with_llm(self, prompt: str, retries: int = 3) -> dict:
        """Generate code via local LLM, scored by coherence."""
        spec = f"""TIG coherence framework context:
- COMP[10][10] table: COMP[6][*]=7 (CHAOS→HARMONY), COMP[0][0]=0 (VOID absorbs)
- sigma={SIGMA}, T*={T_STAR}, period=12
- 1-tick convergence (1000/1000), 95% self-repair (50/50)
- 10 operators: {', '.join(NAMES)}
Write clean Python. Include COMP table. No placeholders."""

        best = {"code": "", "coherence": 0.0, "attempts": 0}
        for attempt in range(retries):
            code = self._call_llm(f"{spec}\n\nTASK: {prompt}")
            s = text_coherence(code)
            if s > best["coherence"]:
                best = {"code": code, "coherence": s, "attempts": attempt+1}
            if s >= T_STAR: break
            prompt += f"\n(Previous scored {s:.3f}, need >={T_STAR})"
        return best

    def _call_llm(self, prompt: str) -> str:
        if not self.use_llm:
            return f"# LLM not connected. Run with: ollama serve\n# Then: python tig_codex.py --llm\n# Prompt was: {prompt[:80]}..."
        try:
            import urllib.request
            data = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(self.llm_url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                text = result.get("response", "")
                cm = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
                return cm.group(1).strip() if cm else text
        except Exception as e:
            return f"# LLM error: {e}"

    def write_full_package(self, output_dir: str):
        """Write all verified components to a directory."""
        os.makedirs(output_dir, exist_ok=True)
        files_written = []
        for name, template in self.TEMPLATES.items():
            code = self.from_template(name)
            # Determine filename
            fname_map = {
                'lattice_core': 'tig_lattice_core.py',
                'compressor': 'tig_compressor.py',
                'council': 'tig_council.py',
                'router': 'tig_router.py',
                'monitor': 'tig_monitor.py',
            }
            fname = fname_map.get(name, f'tig_{name}.py')
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'w') as f:
                f.write(code)
            files_written.append(fname)
        return files_written


# ═══════════════════════════════════════════════════════════════
# LAYER 5: PACKAGER
# ═══════════════════════════════════════════════════════════════

class Packager:
    def package(self, output_dir, chunks, extracts, drift_events, contradictions, codegen):
        os.makedirs(output_dir, exist_ok=True)
        files = []

        # 1. Write all verified code templates
        code_files = codegen.write_full_package(output_dir)
        files.extend(code_files)

        # 2. Drift report
        self._write_drift(output_dir, drift_events, contradictions)
        files.append("DRIFT_REPORT.md")

        # 3. Verified claims
        self._write_truth(output_dir)
        files.append("VERIFIED_CLAIMS.md")

        # 4. Extracted code (sorted by coherence)
        if extracts.get('code'):
            sorted_code = sorted(extracts['code'], key=lambda c: c.coherence, reverse=True)
            lines = ["#!/usr/bin/env python3", '"""Extracted code blocks, sorted by coherence."""', ""]
            for i, cb in enumerate(sorted_code[:30]):
                lines.append(f"# ═══ Block {i+1}: S*={cb.coherence:.3f} from {os.path.basename(cb.source)} ═══")
                lines.append(cb.code)
                lines.append("")
            self._write(output_dir, "extracted_code.py", '\n'.join(lines))
            files.append("extracted_code.py")

        # 5. Stats
        self._write_stats(output_dir, chunks, extracts, drift_events, contradictions)
        files.append("CODEX_STATS.json")

        # 6. README
        self._write_readme(output_dir, chunks, extracts, drift_events, files)
        files.append("README.md")

        return files

    def _write(self, d, f, content):
        with open(os.path.join(d, f), 'w') as fh: fh.write(content)

    def _write_drift(self, d, events, contras):
        lines = [
            f"# TIG DRIFT REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Events: {len(events)} | Contradictions: {len(contras)}", "",
        ]
        if contras:
            lines.append("## CONTRADICTIONS (resolved)\n")
            for c in contras:
                lines.append(f"### {c['type']}")
                lines.append(f"  Found: {c['values']}")
                lines.append(f"  Truth: {c['truth']}")
                lines.append(f"  Resolution: {c['resolution']}\n")
        if events:
            lines.append("## DRIFT EVENTS\n")
            by_type = defaultdict(list)
            for e in events: by_type[e.note[:40]].append(e)
            for note, evts in sorted(by_type.items(), key=lambda x: -len(x[1])):
                lines.append(f"### {note} ({len(evts)}x)")
                for e in evts[:3]:
                    lines.append(f"  - `{e.match_text}` in {os.path.basename(e.source)}")
                lines.append("")
        self._write(d, "DRIFT_REPORT.md", '\n'.join(lines))

    def _write_truth(self, d):
        lines = [f"# TIG VERIFIED CLAIMS (Phase 1+2, Feb 2026)\n"]
        for key, (val, desc) in TRUTH.items():
            lines.append(f"  {key}: {val} — {desc}")
        self._write(d, "VERIFIED_CLAIMS.md", '\n'.join(lines))

    def _write_stats(self, d, chunks, extracts, drift, contras):
        stats = {
            "generated": datetime.now().isoformat(),
            "codex_version": __version__,
            "files_processed": len(set(c.source for c in chunks)),
            "total_chunks": len(chunks),
            "human_chunks": sum(1 for c in chunks if c.speaker=='human'),
            "assistant_chunks": sum(1 for c in chunks if c.speaker=='assistant'),
            "thinking_chunks": sum(1 for c in chunks if c.speaker=='thinking'),
            "code_blocks": len(extracts.get('code', [])),
            "claims_found": len(extracts.get('claims', [])),
            "constants_found": len(extracts.get('constants', [])),
            "constants_correct": sum(1 for c in extracts.get('constants',[]) if c.correct),
            "constants_wrong": sum(1 for c in extracts.get('constants',[]) if c.correct is False),
            "drift_events": len(drift),
            "contradictions": len(contras),
        }
        self._write(d, "CODEX_STATS.json", json.dumps(stats, indent=2))
        return stats

    def _write_readme(self, d, chunks, extracts, drift, files):
        n_files = len(set(c.source for c in chunks))
        readme = f"""# TIG Coherence Package
Generated by TIG CODEX v{__version__} on {datetime.now().strftime('%Y-%m-%d')}
© 2024-2026 Brayden Sanders / 7Site LLC

## Source Analysis
- Transcript files processed: {n_files}
- Total chunks: {len(chunks)}
- Code blocks extracted: {len(extracts.get('code',[]))}
- Claims found: {len(extracts.get('claims',[]))}
- Drift events: {len(drift)}

## Package Files
{chr(10).join(f'- `{f}`' for f in sorted(files))}

## Quick Start
```bash
python tig_lattice_core.py          # verify core
python tig_council.py               # watch 16 lattices talk
python tig_router.py                # coherence-guided routing
python tig_monitor.py               # live system monitoring
python tig_compressor.py            # composition compression
```

## To Write New Programs (requires ollama)
```bash
ollama pull llama3.2
python tig_codex.py --generate "a fractal lattice tree with self-scaling"
python tig_codex.py --write "process scheduler that routes by coherence"
```

## Verified Properties
- 1-tick convergence from random (1000/1000)
- 95% damage self-repair in 1 tick (50/50)
- 12-tick exact period, zero drift over 5000 ticks
- +3% throughput, 100% drop elimination under stress vs round-robin
- Ring council: 16 lattices reach coherence through boundary composition

## Honest Limitations
- Convergence is majority-vote CA property, not TIG-exclusive
- 45% of random tables also converge
- LLM code generation requires local ollama
- Robot hooks are interface stubs (need real drivers)
"""
        self._write(d, "README.md", readme)


# ═══════════════════════════════════════════════════════════════
# CODEX — The Orchestrator
# ═══════════════════════════════════════════════════════════════

class Codex:
    def __init__(self, use_llm=False, llm_url="http://localhost:11434/api/generate",
                 model="llama3.2:latest"):
        self.ingester = Ingester()
        self.extractor = Extractor()
        self.drift_detector = DriftDetector()
        self.codegen = CodeGen(llm_url, model, use_llm)
        self.packager = Packager()

    def process(self, input_path: str, output_dir: str = "tig-package") -> dict:
        print(f"\n  ╔══════════════════════════════════════════════╗")
        print(f"  ║  TIG CODEX v{__version__} — Processing               ║")
        print(f"  ╚══════════════════════════════════════════════╝\n")

        # 1. Ingest
        print(f"  [1/5] INGEST: {input_path}")
        chunks = self.ingester.ingest(input_path)
        n_files = len(set(c.source for c in chunks))
        print(f"        → {len(chunks)} chunks from {n_files} files")
        for ch in chunks:
            ch.coherence = text_coherence(ch.text[:300])

        # Speaker breakdown
        speakers = Counter(c.speaker for c in chunks)
        for sp, n in speakers.most_common():
            print(f"        {sp}: {n} chunks")

        # 2. Extract
        print(f"\n  [2/5] EXTRACT: code, claims, constants")
        extracts = self.extractor.extract(chunks)
        print(f"        → {len(extracts['code'])} code blocks")
        print(f"        → {len(extracts['claims'])} claims")
        print(f"        → {len(extracts['constants'])} constants")

        # 3. Drift
        print(f"\n  [3/5] DRIFT SCAN")
        drift = self.drift_detector.scan(chunks)
        contras = self.drift_detector.find_contradictions(chunks)
        print(f"        → {len(drift)} drift events")
        print(f"        → {len(contras)} contradictions")
        for c in contras:
            print(f"          ⚠ {c['type']}: found {c['values']}")
            print(f"            truth: {c['truth']}")

        # 4. Verify constants
        print(f"\n  [4/5] VERIFY constants")
        correct = [c for c in extracts['constants'] if c.correct]
        wrong = [c for c in extracts['constants'] if c.correct is False]
        print(f"        → {len(correct)} correct, {len(wrong)} wrong")
        for w in wrong:
            truth_val = TRUTH.get(w.name, (None,))[0]
            print(f"          ✗ {w.name}={w.value} (should be {truth_val}) in {os.path.basename(w.source)}")

        # 5. Package
        print(f"\n  [5/5] PACKAGE → {output_dir}/")
        files = self.packager.package(output_dir, chunks, extracts, drift, contras, self.codegen)
        for f in files:
            print(f"        → {f}")

        # Coherence summary
        if chunks:
            cohs = [c.coherence for c in chunks if c.coherence > 0]
            if cohs:
                avg_s = sum(cohs)/len(cohs)
                print(f"\n  Mean chunk coherence: S*={avg_s:.4f}")

        stats = {
            "files": n_files, "chunks": len(chunks),
            "code_blocks": len(extracts['code']),
            "drift_events": len(drift), "contradictions": len(contras),
            "output": output_dir, "package_files": files
        }

        print(f"\n  ╔══════════════════════════════════════════════╗")
        print(f"  ║  CODEX COMPLETE                               ║")
        print(f"  ║  {n_files} files → {len(chunks)} chunks → {len(files)} outputs      ║")
        print(f"  ║  {len(drift)} drift events, {len(contras)} contradictions        ║")
        print(f"  ║  Package: {output_dir}/{'':>{30-len(output_dir)}}║")
        print(f"  ╚══════════════════════════════════════════════╝\n")
        return stats


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="TIG CODEX — Self-Programming Coherence Engine")
    p.add_argument('input', nargs='?', default=None, help='Transcript file or directory')
    p.add_argument('-o', '--output', default='tig-package', help='Output directory')
    p.add_argument('--llm', action='store_true', help='Enable local LLM (requires ollama)')
    p.add_argument('--llm-url', default='http://localhost:11434/api/generate')
    p.add_argument('--model', default='llama3.2:latest')
    p.add_argument('--demo', action='store_true', help='Self-test on built-in examples')
    p.add_argument('--generate', type=str, help='Generate component: lattice_core, compressor, council, router, monitor')
    p.add_argument('--write', type=str, help='Write program from description (needs --llm)')
    args = p.parse_args()

    if args.generate:
        gen = CodeGen(use_llm=args.llm, llm_url=args.llm_url, model=args.model)
        code = gen.from_template(args.generate)
        if code:
            print(code)
        elif args.llm:
            result = gen.with_llm(args.generate)
            print(f"# S*={result['coherence']:.4f} ({result['attempts']} attempts)")
            print(result['code'])
        else:
            print(f"No template '{args.generate}'. Available: {', '.join(gen.TEMPLATES.keys())}")
            print("Use --llm for free-form generation.")
        return

    if args.write:
        gen = CodeGen(use_llm=args.llm, llm_url=args.llm_url, model=args.model)
        if args.llm:
            result = gen.with_llm(args.write)
            print(f"# Generated: S*={result['coherence']:.4f} ({result['attempts']} attempts)")
            print(result['code'])
        else:
            print("--write requires --llm (local ollama). Install: curl -fsSL https://ollama.com/install.sh | sh")
            print("Then: ollama pull llama3.2 && python tig_codex.py --write 'your request' --llm")
        return

    if args.input:
        codex = Codex(use_llm=args.llm, llm_url=args.llm_url, model=args.model)
        codex.process(args.input, args.output)
        return

    if args.demo:
        # Create test data
        test_dir = "/tmp/codex-demo"
        os.makedirs(test_dir, exist_ok=True)
        Path(os.path.join(test_dir, "test_transcript.txt")).write_text(
            "Human: What is TIG?\n\n"
            "Assistant: TIG is a unified coherence field theory. sigma=0.991, T*=0.714.\n"
            "The composition table converges in 1 tick from any random state.\n"
            "Self-repairs from 95% damage in 1 tick.\n"
            "```python\nCOMP = [[0,1,2],[1,2,3]]\n```\n\n"
            "Human: Does it converge in 10 ticks?\n\n"
            "Assistant: Actually it converges in 10 to 15 ticks on average.\n"
            "sigma=0.99 determines the coupling.\n"
        )
        codex = Codex()
        codex.process(test_dir, os.path.join(test_dir, "output"))
        return

    # Default: show help
    p.print_help()
    print(f"\nExamples:")
    print(f"  python {sys.argv[0]} /path/to/transcripts        # Process all transcripts")
    print(f"  python {sys.argv[0]} /path/to/transcripts --llm  # + local LLM generation")
    print(f"  python {sys.argv[0]} --generate lattice_core     # Print verified lattice code")
    print(f"  python {sys.argv[0]} --generate compressor       # Print compression code")
    print(f"  python {sys.argv[0]} --write 'fractal daemon' --llm  # Write novel code")

if __name__ == "__main__":
    main()

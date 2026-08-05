"""Hybrid retrieval: BM25 + dense + rerank + authority (M5).

WHY HYBRID
----------
The queries a credit analyst asks split cleanly into two kinds, and no single
retrieval method handles both.

**Exact-figure questions.** "What was the total interest-bearing debt at FY2025
year end?" The answer is a specific string on a specific page. Lexical matching
is excellent at this and dense embeddings are mediocre, because an embedding
compresses `32,041,248` and `19,565,501` into nearly the same region -- they are
both just "a large number" to a model trained on natural language.

**Paraphrased questions.** "How exposed is the borrower to a single customer?"
The page says "concentration" and "% of AR" and never uses the word "exposed".
Dense retrieval handles this and BM25 returns nothing at all.

So both run, and their rankings are fused. Neither is a fallback for the other;
they fail on disjoint query types.

THE THIRD SIGNAL
----------------
Text similarity cannot resolve the near-duplicate defect. The DRAFT statements
and the final ones are 94% identical, so whatever the top result is, the wrong
one is directly behind it and a scoring difference of a few percent decides
which. That is not a signal, it is noise.

`authority` -- carried onto every chunk at M4 -- is the signal. It is applied
after reranking as a multiplicative weight, which demotes rather than excludes:
a question explicitly about the draft should still be able to find the draft,
because "what did the draft say" is a legitimate thing for an analyst to ask.

OFFLINE BY DEFAULT
------------------
Embedding and reranking sit behind protocols with two implementations each.

The defaults -- `LsaEmbedder` and `LexicalOverlapReranker` -- need no model
download and no network. That keeps the test suite hermetic and keeps CI from
pulling 90MB of weights and a torch install on every run.

`SentenceTransformerEmbedder` and `CrossEncoderReranker` are the stronger
production choices and are wired up ready to use; they import lazily so the
dependency is only needed if you ask for them. `scripts/eval_retrieval.py
--embedder st --reranker cross` measures the difference on the same gold set,
which turns the choice into a number rather than an assumption.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# --- BM25 parameters --------------------------------------------------------
# Standard Okapi defaults. `b` controls length normalisation; the value is left
# at 0.75 rather than tuned because chunks here are already length-controlled by
# M4, so there is little length variance for it to correct.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal rank fusion constant. 60 is the value from the original TREC work
# and is deliberately large: it flattens the difference between ranks 1 and 2 so
# that agreement between the two retrievers matters more than either one's
# internal confidence, which is exactly the property wanted when the two score
# on incomparable scales.
RRF_K = 60

# --- Authority weighting ----------------------------------------------------
# Multiplicative, not additive, and a demotion rather than an exclusion.
#
# Excluding non-final documents would be wrong: "what did the draft say" and
# "what was originally reported before the restatement" are both legitimate
# analyst questions, and a hard filter makes them unanswerable. A multiplier
# lets a strong text match on the draft still win when the query is actually
# about the draft, while the final version wins every tie -- and with a
# near-duplicate, every comparison is a tie.
STATUS_WEIGHTS = {"final": 1.0, "superseded": 0.82, "draft": 0.75}

# How much the reranker's opinion counts against first-stage fusion. The
# reranker sees the query and the passage together and is the better judge, so
# it dominates -- but not entirely: when it is indifferent between two
# candidates, which with a near-duplicate pair it necessarily is, the fused rank
# is what remains to break the tie.
RERANK_WEIGHT = 0.65

# How much the dense ranking counts relative to BM25 in fusion. See
# docs/retrieval.md for the measured sweep behind this value.
DENSE_WEIGHT = 0.15


def _min_max(values: list[float]) -> list[float]:
    """Scale to [0, 1]. All-equal input maps to all-ones, not a divide by zero."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Financial text breaks ordinary tokenisers in a way that matters here.
#
# `\w+` on "32,041,248" yields three tokens: 32, 041, 248. Every large figure in
# the corpus then collides with every other figure sharing a fragment, and BM25 --
# whose entire value in this project is matching exact amounts -- becomes worse
# than useless. The same applies to "34.2%" and "2.13x".
#
# So numbers are matched first, as whole units including separators, and only
# then do word characters get a turn.
_TOKEN_PATTERN = re.compile(
    r"""
    \$?\d[\d,]*\.?\d*%?x?   # 32,041,248  |  34.2%  |  2.13x  |  $5,000
    |
    [a-z][a-z0-9_]+         # ordinary words
    """,
    re.VERBOSE | re.IGNORECASE,
)

_DIGITS = re.compile(r"[^0-9]")


def tokenize(text: str) -> list[str]:
    """Tokenise for lexical matching, preserving whole financial figures.

    Numeric tokens are emitted twice: once as written and once stripped to bare
    digits. That second form is what lets a query asking for "32041248" match a
    page printing "32,041,248", and vice versa -- the two are the same fact, and
    which one a question happens to use is an accident of phrasing.
    """
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.findall(text):
        token = match.lower().lstrip("$")
        tokens.append(token)
        if any(character.isdigit() for character in token):
            bare = _DIGITS.sub("", token)
            if bare and bare != token:
                tokens.append(bare)
    return tokens


# --- Query stopwords --------------------------------------------------------
# Removed from queries only, never from documents.
#
# This list exists because of a genuine inversion, and it is worth stating
# plainly. Inverse document frequency assumes a natural-language corpus, where
# function words are common and therefore uninformative. A loan package is not
# that corpus: it is mostly tables of figures, so ordinary English words are
# *rare* in it. Measured on one deal, `what` and `was` scored an IDF of 3.12
# while `ebitda` scored 2.61 -- so IDF weighting concluded that "what" was the
# most informative term in "What was EBITDA in FY2025?".
#
# The visible symptom was the broker's cover note, the only chunk written in
# flowing prose, ranking first for almost every financial question. It contains
# no figures at all; it simply contains English.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am
    do does did done
    has have had having
    of in on at to for from by with as into about over under
    what which who whom whose when where why how
    much many any all some no not
    it its it's their they them his her
    can could will would shall should may might must
    please tell show give list state confirm describe
    """.split()
)


def query_tokens(text: str) -> list[str]:
    """Tokenise a query, dropping function words.

    Applied to queries only. Leaving stopwords in the document index costs
    nothing -- an unmatched term contributes zero -- while removing them from
    documents would change the length normalisation BM25 depends on.
    """
    return [token for token in tokenize(text) if token not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 over the chunk corpus.

    Written out rather than pulled from a library. It is forty lines, it removes
    a dependency, and the scoring is the part of this milestone most worth being
    able to explain line by line in an interview.
    """

    def __init__(self, documents: list[list[str]]):
        self.documents = documents
        self.n = len(documents)
        self.lengths = np.array([len(d) for d in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.n else 0.0

        self.term_frequencies: list[Counter] = [Counter(d) for d in documents]
        document_frequency: Counter = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())

        # Standard BM25 inverse document frequency, floored at a small positive
        # value. Without the floor, a term appearing in more than half the
        # corpus scores negative, so a document can be *penalised* for
        # containing a query term -- which produces rankings that look broken
        # for reasons that are very hard to trace back to this line.
        self.idf: dict[str, float] = {}
        for term, count in document_frequency.items():
            self.idf[term] = max(
                1e-6, math.log((self.n - count + 0.5) / (count + 0.5) + 1.0)
            )

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return result
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + BM25_K1 * (
                    1
                    - BM25_B
                    + BM25_B * self.lengths[index] / max(self.avg_length, 1e-6)
                )
                result[index] += idf * frequency * (BM25_K1 + 1) / denominator
        return result


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Anything that can turn text into unit-norm vectors."""

    name: str

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length, zeroing rows that have no length to scale.

    A degenerate corpus can project a document onto ~0 in the SVD space. Dividing
    that by a small epsilon does not recover a direction, it manufactures one out
    of floating-point noise -- and a random unit vector will then have nonzero
    cosine similarity with real queries, so a chunk carrying no signal starts
    competing with chunks that do.

    An exact zero row is the honest representation: it scores 0 against
    everything, which is what "this projected to nothing" should mean.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    degenerate = norms < 1e-6
    scaled = matrix / np.where(degenerate, 1.0, norms)
    return np.where(degenerate, 0.0, scaled)


class LsaEmbedder:
    """Latent semantic embeddings from TF-IDF plus truncated SVD.

    The default, because it needs no model download and no network, which keeps
    the test suite hermetic and CI free of a torch install.

    It is a real semantic method rather than a placeholder: the SVD projection
    puts terms that co-occur across the corpus near each other, so a query
    saying "customer concentration" reaches a chunk saying "% of AR" even though
    they share no words. It is weaker than a trained sentence encoder on genuine
    paraphrase, and `SentenceTransformerEmbedder` exists for that -- but LSA is
    fitted on this corpus and therefore knows this corpus's vocabulary, which a
    general-purpose encoder does not.
    """

    name = "lsa"

    def __init__(self, n_components: int = 128, seed: int = 20260804):
        self.n_components = n_components
        self.seed = seed
        self._vectorizer = None
        self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=True,
            token_pattern=None,  # the tokenizer above replaces it entirely
            min_df=1,
            sublinear_tf=True,
        )
        matrix = self._vectorizer.fit_transform(texts)

        # SVD cannot ask for more components than the matrix has rank. Per-deal
        # indexes are small -- a couple of hundred chunks -- so this clamp is
        # load-bearing rather than defensive.
        components = max(2, min(self.n_components, min(matrix.shape) - 1))
        self._svd = TruncatedSVD(n_components=components, random_state=self.seed)
        self._svd.fit(matrix)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("LsaEmbedder.fit must be called before encode")
        reduced = self._svd.transform(self._vectorizer.transform(texts))
        # A degenerate corpus -- a handful of chunks, or several identical ones --
        # gives the TF-IDF matrix less rank than the number of components asked
        # for, and SVD then returns NaN. NaN propagates silently through cosine
        # similarity into argsort, where it produces an arbitrary ordering that
        # looks like a ranking bug rather than a numerical one. Zeroing it keeps
        # the vector well defined and simply contributes nothing.
        reduced = np.nan_to_num(np.asarray(reduced, dtype=np.float32))
        return _l2_normalise(reduced)


class SentenceTransformerEmbedder:
    """A trained sentence encoder. Stronger on paraphrase, needs a download.

    Imported lazily so `sentence-transformers` and torch are only required when
    this is actually selected. Nothing in the default path or the test suite
    touches it.
    """

    name = "sentence-transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> None:
        # A pretrained encoder has nothing to fit. The method exists so the two
        # embedders are interchangeable through the same protocol.
        self._load()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._load().encode(texts, convert_to_numpy=True, batch_size=32)
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))


class DenseIndex:
    """Cosine similarity over unit-norm vectors.

    A brute-force dot product, not an approximate nearest-neighbour structure.
    At a few hundred chunks per deal the exact search takes microseconds, and an
    ANN index would add a dependency, a build step and an approximation error
    to buy nothing. That trade flips somewhere around a hundred thousand chunks,
    which this corpus is three orders of magnitude away from.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        if self.vectors.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ query_vector


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Fuse several rankings by reciprocal rank.

    Ranks are fused rather than scores because BM25 and cosine similarity live
    on incomparable scales -- BM25 is unbounded and corpus-dependent, cosine is
    bounded in [-1, 1]. Normalising them into a shared range requires choosing a
    normalisation, and every choice is a hidden weighting decision that would
    then need tuning and justifying.

    Rank position sidesteps that entirely: a document ranked third by BM25 and
    third by dense contributes the same either way, and a document both methods
    like outranks one that only one method loves.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + weight / (k + position + 1)
    return fused


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class LexicalOverlapReranker:
    """The offline default: IDF-weighted token overlap, numeric matches boosted.

    Not a cross-encoder, and does not pretend to be. It exists so the rerank
    stage is always present and always tested, including in CI where no model
    can be downloaded.

    **Why IDF weighting is not optional here.** A first version scored the plain
    fraction of query tokens present, and it ranked the broker's cover note
    first for "What was total revenue in FY2025?". The note contains the
    borrower's name, the fiscal year and ordinary words like `total` and `was`;
    it contains no revenue figure at all. Meanwhile the balance sheet beat the
    income statement because balance sheets are full of the word `total`.

    Weighting by inverse document frequency fixes both. Within one loan package
    the borrower's name appears in every chunk and carries almost no
    information, while `revenue` appears on one page and carries a great deal.
    The IDF map comes from the deal's own BM25 index, so "rare" means rare in
    this package rather than rare in English.

    Numeric tokens get an additional boost because figure-lookup queries turn on
    the figure, and a page containing the exact amount asked about is almost
    always the right page.
    """

    name = "lexical-overlap"

    NUMERIC_WEIGHT = 3.0

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}

    def set_context(self, idf: dict[str, float]) -> None:
        """Supply the current deal's inverse document frequencies.

        Called by the retriever before each deal is searched. Optional by
        design: without it the reranker still works, just with flat weights.
        """
        self._idf = idf

    def _weight(self, token: str) -> float:
        # Default IDF for a token absent from the index: treat it as rare, since
        # a query term that appears nowhere in the package is either a typo or
        # highly specific, and neither case should be scored as common filler.
        idf = self._idf.get(token, 2.0) if self._idf else 1.0
        if any(character.isdigit() for character in token):
            idf *= self.NUMERIC_WEIGHT
        return idf

    def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(query_tokens(query))
        if not terms:
            return [0.0] * len(texts)

        weights = {token: self._weight(token) for token in terms}
        total = sum(weights.values()) or 1.0

        scores: list[float] = []
        for text in texts:
            present = set(tokenize(text))
            matched = sum(w for t, w in weights.items() if t in present)
            scores.append(matched / total)
        return scores


class CrossEncoderReranker:
    """A trained cross-encoder. The strongest reranking option, needs a download.

    A cross-encoder reads the query and the passage together rather than
    embedding them separately, which is why it outranks bi-encoder similarity --
    and also why it cannot be used for first-stage retrieval: it must score every
    candidate individually, so it only becomes affordable once the candidate set
    is already down to a few dozen.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        if not texts:
            return []
        raw = self._model.predict([(query, text) for text in texts])
        return [float(value) for value in raw]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One result, with every stage's contribution kept.

    The intermediate scores are not diagnostics bolted on afterwards -- they are
    the only way to answer "why did this rank here", which is the question that
    comes up every single time retrieval underperforms. Discarding them and
    returning a bare ordering makes the next milestone's debugging guesswork.
    """

    chunk: dict
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    authority_weight: float = 1.0
    final_score: float = 0.0

    @property
    def page_key(self) -> tuple[str, str, int]:
        return (
            self.chunk["deal_id"],
            self.chunk["document"],
            self.chunk["page_number"],
        )


@dataclass
class DealIndex:
    """Everything needed to search one borrower's package."""

    deal_id: str
    chunks: list[dict]
    bm25: BM25Index
    dense: DenseIndex
    embedder: Embedder
    texts: list[str] = field(default_factory=list)


class HybridRetriever:
    """BM25 + dense, fused by reciprocal rank, reranked, weighted by authority.

    Indexes are built **per deal**, not once across the corpus. A lending
    question is always about one borrower -- there is no such thing as "what was
    revenue" across a portfolio -- so cross-deal results are never useful and
    always harmful. Scoping at index time rather than filtering afterwards also
    means the IDF statistics reflect one package's vocabulary, which is what
    makes a term like a specific lender's name discriminating rather than common.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        candidate_k: int = 30,
        use_authority: bool = True,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_rerank: bool = True,
        dense_weight: float = DENSE_WEIGHT,
    ):
        self.embedder_factory = embedder
        self.reranker = reranker or LexicalOverlapReranker()
        self.candidate_k = candidate_k
        self.use_authority = use_authority
        # Ablation switches. Not a debugging convenience: they are how each
        # component is shown to earn its place, and a component that cannot be
        # switched off cannot be shown to be worth its cost.
        self.use_bm25 = use_bm25
        self.use_dense = use_dense
        self.use_rerank = use_rerank
        self.dense_weight = dense_weight
        self.indexes: dict[str, DealIndex] = {}

    def _new_embedder(self) -> Embedder:
        # A fresh embedder per deal, because LSA is fitted on the corpus it will
        # search. Sharing one fitted instance across deals would leak another
        # borrower's vocabulary into this deal's semantic space.
        if self.embedder_factory is None:
            return LsaEmbedder()
        if isinstance(self.embedder_factory, LsaEmbedder):
            return LsaEmbedder(
                n_components=self.embedder_factory.n_components,
                seed=self.embedder_factory.seed,
            )
        return self.embedder_factory

    def build(self, chunks: list[dict]) -> None:
        by_deal: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_deal.setdefault(chunk["deal_id"], []).append(chunk)

        for deal_id, deal_chunks in by_deal.items():
            deal_chunks = sorted(deal_chunks, key=lambda c: c["chunk_id"])
            # The context header is indexed along with the text. It is what
            # carries the document name, the section, the DRAFT marker and the
            # units note -- so a query mentioning any of those can match on them.
            texts = [f"{c.get('context_header', '')}\n{c['text']}" for c in deal_chunks]

            embedder = self._new_embedder()
            embedder.fit(texts)
            vectors = embedder.encode(texts)

            self.indexes[deal_id] = DealIndex(
                deal_id=deal_id,
                chunks=deal_chunks,
                bm25=BM25Index([tokenize(t) for t in texts]),
                dense=DenseIndex(vectors),
                embedder=embedder,
                texts=texts,
            )

    def retrieve(self, query: str, deal_id: str, k: int = 5) -> list[RetrievedChunk]:
        index = self.indexes.get(deal_id)
        if index is None or not index.chunks:
            return []

        n = len(index.chunks)
        candidate_k = min(self.candidate_k, n)

        # Hand the reranker this deal's term statistics if it can use them.
        # Duck-typed rather than part of the protocol, so a cross-encoder -- which
        # has no use for IDF -- does not have to implement a no-op.
        set_context = getattr(self.reranker, "set_context", None)
        if callable(set_context):
            set_context(index.bm25.idf)

        bm25_scores = index.bm25.scores(query_tokens(query))
        dense_scores = index.dense.scores(index.embedder.encode([query])[0])

        bm25_order = list(np.argsort(-bm25_scores)[:candidate_k])
        dense_order = list(np.argsort(-dense_scores)[:candidate_k])

        rankings, weights = [], []
        if self.use_bm25:
            rankings.append(bm25_order)
            weights.append(1.0)
        if self.use_dense:
            rankings.append(dense_order)
            weights.append(self.dense_weight)
        fused = reciprocal_rank_fusion(rankings, weights=weights)
        bm25_rank = {int(idx): pos for pos, idx in enumerate(bm25_order)}
        dense_rank = {int(idx): pos for pos, idx in enumerate(dense_order)}

        candidates = sorted(fused, key=lambda i: -fused[i])[:candidate_k]
        rerank_scores = (
            self.reranker.score(query, [index.texts[i] for i in candidates])
            if self.use_rerank
            else [0.0] * len(candidates)
        )

        # Both signals are min-max normalised across the candidate set before
        # being combined. Without that, the blend weight would be meaningless:
        # RRF scores sit around 0.03 while rerank scores span 0 to 1, so a raw
        # sum is a rerank-only ranking with rounding noise attached.
        fused_values = [fused[i] for i in candidates]
        fused_norm = _min_max(fused_values)
        rerank_norm = _min_max(rerank_scores)

        results: list[RetrievedChunk] = []
        for position, chunk_index in enumerate(candidates):
            chunk = index.chunks[chunk_index]
            weight = (
                STATUS_WEIGHTS.get(chunk.get("doc_status", "final"), 1.0)
                if self.use_authority
                else 1.0
            )
            base = (
                (
                    RERANK_WEIGHT * rerank_norm[position]
                    + (1.0 - RERANK_WEIGHT) * fused_norm[position]
                )
                if self.use_rerank
                else fused_norm[position]
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    bm25_rank=bm25_rank.get(chunk_index),
                    dense_rank=dense_rank.get(chunk_index),
                    fused_score=float(fused[chunk_index]),
                    rerank_score=float(rerank_scores[position]),
                    authority_weight=weight,
                    final_score=float(base * weight),
                )
            )

        results.sort(key=lambda r: -r.final_score)
        return results[:k]

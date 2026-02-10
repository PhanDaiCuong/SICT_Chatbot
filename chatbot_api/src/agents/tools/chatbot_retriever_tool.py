"""
Retrieval and ranking utilities for building an optimized RAG pipeline.

Overview
--------
- Combines semantic search (Qdrant Vector Store) with lexical search (BM25).
- Fuses results via `EnsembleRetriever` with configurable weights.
- Limits the number of candidates before expensive re-ranking to save compute.
- Re-ranks with a HuggingFace Cross-Encoder, then returns compressed, relevant documents.
- Persists BM25 index to disk for faster subsequent startup.

Key Components
--------------
- `BM25IndexManager`: Loads/builds a BM25 index and persists it via pickle.
- `LimitRetriever`: Thin wrapper limiting number of docs prior to re-ranking.
- `build_optimized_rag_pipeline(...)`: Composes the full pipeline end-to-end.

Safety & Notes
--------------
- Pickle persistence is version- and environment-sensitive. Rebuild the index
  if you update dependencies or document structure (`force_rebuild_bm25=True`).
- `_get_device()` currently returns "cpu"; adapt if you want GPU acceleration.

Quick Usage
-----------
```
final_retriever = build_optimized_rag_pipeline(
    vector_store=qdrant_store,
    bm25_manager=BM25IndexManager("bm25_index.pkl"),
    documents_for_bm25=docs,  # only required when building the BM25 index
)
results = final_retriever.invoke("What is the tuition fee?")
```
"""

import os
import logging
import pickle
from typing import List, Optional, Any
from pydantic import Field, ConfigDict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker



# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Pipeline")


def _get_device() -> str:
    """Return the compute device to use for the cross-encoder.

    Notes
    -----
    - Currently hardcoded to "cpu" to avoid dependency on local GPU setup.
    - If you have CUDA available and want GPU acceleration, replace this
      implementation with detection logic (e.g., using `torch.cuda.is_available`).

    Returns
    -------
    str
        The device identifier; currently "cpu".
    """
    return 'cpu'


class BM25IndexManager:
    """Manage BM25 index persistence (load/build/save) via pickle.

    Parameters
    ----------
    index_path : str
        Path to the pickle file where the BM25 index is stored.

    Notes
    -----
    - Pickle files are not portable across all environments; changes in
      dependency versions may require a rebuild.
    - Ensure the documents used to build the index are representative of the
      corpus you intend to search.
    """
    
    def __init__(self, index_path: str = "bm25_index.pkl"):
        self.index_path = index_path

    def load_or_build(self, documents: List[Document] = None, force_rebuild: bool = False) -> BM25Retriever:
        """Load an existing BM25 index or build a new one.

        Parameters
        ----------
        documents : List[Document], optional
            Documents to build the BM25 index from. Required when building.
        force_rebuild : bool, default False
            If True, rebuild the index even if a pickle file exists.

        Returns
        -------
        BM25Retriever
            A configured retriever ready for lexical search.

        Raises
        ------
        ValueError
            If `documents` is None or empty when building is required.
        """
        # 1. Thử load từ file
        if os.path.exists(self.index_path) and not force_rebuild:
            try:
                with open(self.index_path, "rb") as f:
                    logger.info(f" Loading BM25 index from {self.index_path}...")
                    retriever = pickle.load(f)
                    if not isinstance(retriever, BM25Retriever):
                        raise ValueError("File content is not a BM25Retriever")
                    return retriever
            except Exception as e:
                logger.warning(f"Failed to load BM25 index ({e}). Rebuilding...")

        # 2. Build mới
        if not documents:
            raise ValueError("Cannot build BM25: 'documents' list is empty or None!")
        
        logger.info(f"🔨 Building BM25 index with {len(documents)} documents...")
        retriever = BM25Retriever.from_documents(documents)
        
        # 3. Lưu xuống đĩa
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(retriever, f)
            logger.info(f"BM25 index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Could not save BM25 index: {e}")
        
        return retriever


class LimitRetriever(BaseRetriever):
    """
    Wrapper retriever limiting the number of candidate documents before re-ranking.

    This helps reduce computational cost for cross-encoder stages by capping
    the number of inputs to re-ranking.

    Attributes
    ----------
    source_retriever : BaseRetriever
        The underlying retriever providing initial candidates (e.g., hybrid).
    limit : int
        Maximum number of documents to pass to the downstream compressor/reranker.
    """
    source_retriever: BaseRetriever
    limit: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Retrieve documents from the source retriever and truncate to `limit`.

        Parameters
        ----------
        query : str
            The user query to search.
        run_manager : CallbackManagerForRetrieverRun
            LangChain callback manager for the retriever run.

        Returns
        -------
        List[Document]
            At most `limit` documents from the underlying retriever.
        """
        docs = self.source_retriever.invoke(query)
        return docs[:self.limit]


def build_optimized_rag_pipeline(
    vector_store: Any,  
    bm25_manager: BM25IndexManager,
    documents_for_bm25: Optional[List[Document]] = None,
    force_rebuild_bm25: bool = False,
    # --- TUNING ---
    k_semantic: int = 40,
    k_bm25: int = 40,
    fusion_top_k: int = 60,
    rerank_top_n: int = 7,
    ensemble_weights: List[float] = [0.4, 0.6],
    cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> BaseRetriever:
    """Compose a hybrid + re-ranking RAG retriever with sensible defaults.

    Pipeline
    --------
    1. Semantic search via the provided `vector_store` (`k_semantic` results).
    2. Lexical search via BM25 (`k_bm25` results), loaded/built by `bm25_manager`.
    3. Weighted fusion with `EnsembleRetriever`.
    4. Candidate limiting to `fusion_top_k`.
    5. Cross-encoder re-ranking to top `rerank_top_n` documents.

    Parameters
    ----------
    vector_store : Any
        A LangChain vector store exposing `.as_retriever(...)` and a client.
    bm25_manager : BM25IndexManager
        Manager responsible for loading or building the BM25 index.
    documents_for_bm25 : List[Document], optional
        Documents to build BM25; required if the index needs to be created.
    force_rebuild_bm25 : bool, default False
        Rebuild the BM25 index regardless of existing pickle file.
    k_semantic : int, default 40
        Number of candidates from the semantic layer.
    k_bm25 : int, default 40
        Number of candidates from the lexical layer.
    fusion_top_k : int, default 60
        Cap on candidates forwarded to the cross-encoder stage.
    rerank_top_n : int, default 7
        Number of final documents after re-ranking.
    ensemble_weights : List[float], default [0.4, 0.6]
        Relative weights for semantic vs lexical components.
    cross_model_name : str, default "cross-encoder/ms-marco-MiniLM-L-6-v2"
        HuggingFace cross-encoder model name for re-ranking.

    Returns
    -------
    BaseRetriever
        A `ContextualCompressionRetriever` performing hybrid retrieval and
        cross-encoder re-ranking.

    Raises
    ------
    ValueError
        If BM25 must be built but `documents_for_bm25` is missing/empty.

    Examples
    --------
    ````python
    retriever = build_optimized_rag_pipeline(
        vector_store=vstore,
        bm25_manager=BM25IndexManager("bm25.pkl"),
        documents_for_bm25=docs,
        k_semantic=30,
        k_bm25=30,
        fusion_top_k=50,
        rerank_top_n=5,
    )
    results = retriever.invoke("Latest admission schedule")
    ````
    """
    
    logger.info("Initializing RAG Pipeline Components...")

    # 1. Semantic Layer (Qdrant)
    qdrant_retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k_semantic}
    )

    # 2. Lexical Layer (BM25)
    bm25_retriever = bm25_manager.load_or_build(
        documents=documents_for_bm25, 
        force_rebuild=force_rebuild_bm25
    )
    bm25_retriever.k = k_bm25

    # 3. Ensemble Layer (Hybrid Search)
    base_ensemble = EnsembleRetriever(
        retrievers=[qdrant_retriever, bm25_retriever],
        weights=ensemble_weights 
    )

    # 4. Limit Layer (Pre-Rerank Optimization)
    limited_retriever = LimitRetriever(
        source_retriever=base_ensemble,
        limit=fusion_top_k 
    )

    # 5. Rerank Layer (Cross-Encoder)
    device = _get_device()
    logger.info(f"Reranker ({cross_model_name}) running on: {device.upper()}")
    
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=cross_model_name,
        model_kwargs={'device': device}
    )
    
    reranker_compressor = CrossEncoderReranker(model=cross_encoder, top_n=rerank_top_n)
    
    # 6. Final Pipeline
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker_compressor,
        base_retriever=limited_retriever
    )
    
    logger.info(
        f"Pipeline Ready: [Qdrant(k={k_semantic}) + BM25(k={k_bm25})] "
        f"-> Top({fusion_top_k}) -> Rerank({device}) -> Final({rerank_top_n})"
    )
    
    return final_retriever
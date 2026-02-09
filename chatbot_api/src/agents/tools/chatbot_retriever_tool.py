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
    """Hàm phụ trợ để tự động phát hiện GPU/CPU an toàn."""
    return 'cpu'


class BM25IndexManager:
    """Quản lý lưu/tải BM25 Index sử dụng Pickle."""
    
    def __init__(self, index_path: str = "bm25_index.pkl"):
        self.index_path = index_path

    def load_or_build(self, documents: List[Document] = None, force_rebuild: bool = False) -> BM25Retriever:
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
    Retriever Wrapper để giới hạn số lượng documents TRƯỚC khi đưa vào Reranker.
    Giúp giảm tải cho Cross-Encoder.
    """
    source_retriever: BaseRetriever
    limit: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
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
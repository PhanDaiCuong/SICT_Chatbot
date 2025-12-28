import os
import logging
import pickle
from typing import List, Optional
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.schema import Document, BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


# Cấu hình log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Pipeline")

class BM25IndexManager:
    def __init__(self, index_path: str = "bm25_index.pkl"):
        self.index_path = index_path

    def load_or_build(self, documents: List[Document] = None, force_rebuild: bool = False) -> BM25Retriever:
        # 1. Nếu file tồn tại và KHÔNG ép build lại -> Load
        if os.path.exists(self.index_path) and not force_rebuild:
            try:
                with open(self.index_path, "rb") as f:
                    logger.info(f"📂 Loading BM25 index from {self.index_path}...")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load BM25 index: {e}. Rebuilding...")

        # 2. Build mới
        if not documents:
            raise ValueError("Cần cung cấp documents để build BM25 index mới!")
        
        logger.info(f"🔨 Building BM25 index with {len(documents)} documents...")
        retriever = BM25Retriever.from_documents(documents)
        
        # Lưu xuống đĩa
        with open(self.index_path, "wb") as f:
            pickle.dump(retriever, f)
        logger.info(f"✅ BM25 index saved/updated to {self.index_path}")
        
        return retriever



class LimitRetriever(BaseRetriever):
    source_retriever: BaseRetriever
    limit: int 

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Lấy full danh sách từ Ensemble
        docs = self.source_retriever.invoke(query, config={"callbacks": run_manager})
        # Cắt lát
        return docs[:self.limit]

def build_optimized_rag_pipeline(
    vector_store: Qdrant,
    bm25_manager: BM25IndexManager,
    documents_for_bm25: Optional[List[Document]] = None,
    force_rebuild_bm25: bool = False,
    # --- CẤU HÌNH SỐ LƯỢNG (TUNING) ---
    k_semantic: int = 20,    # Tăng lên để tăng khả năng tìm thấy (Recall)
    k_bm25: int = 20,        # Tăng lên
    fusion_top_k: int = 30,  # QUAN TRỌNG: Lấy top 30 để Reranker có cái mà chọn
    rerank_top_n: int = 5,   # Kết quả cuối cùng cho LLM
    cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    
    # 1. Semantic
    qdrant_retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k_semantic}
    )

    bm25_retriever = bm25_manager.load_or_build(
        documents=documents_for_bm25, 
        force_rebuild=force_rebuild_bm25
    )
    bm25_retriever.k = k_bm25

    # 3. Ensemble
    # Semantic 0.5, Lexical 0.5 là khởi điểm an toàn nhất
    base_ensemble = EnsembleRetriever(
        retrievers=[qdrant_retriever, bm25_retriever],
        weights=[0.5, 0.5] 
    )

    # 4. Limit (Cắt lát)
    limited_retriever = LimitRetriever(
        source_retriever=base_ensemble,
        limit=fusion_top_k 
    )

    # 5. Reranker Logic
    use_gpu_env = os.getenv('USE_GPU', 'False').lower() in ('true', '1', 't')
    device = 'cuda' if use_gpu_env else 'cpu'

    logger.info(f"⚙️ Reranker running on: {device.upper()}")
    
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=cross_model_name,
        model_kwargs={'device': device}
    )
    
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=rerank_top_n)
    
    # 6. Final Compression
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=limited_retriever
    )
    
    logger.info(f"🚀 Pipeline: (Qdrant={k_semantic} + BM25={k_bm25}) -> Top {fusion_top_k} -> Rerank -> Top {rerank_top_n}")
    return final_retriever
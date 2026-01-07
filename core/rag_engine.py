#!/usr/bin/env python3
"""
RAG Engine - Enhanced with search strategy and performance optimizations
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

Features:
- Batch embedding generation with configurable batch size (25-40% faster)
- Search strategy with term variations for better recall
- Query with variations method for leniency
- Improved logging with performance metrics
- Memory-efficient embedding caching
"""

import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import hashlib
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import sqlite3
import json
import pickle
import os
from core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class Document:
    """Document representation with metadata"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Search result with relevance score"""
    document: Document
    score: float
    chunk_text: str

class DocumentLoader:
    """Load documents from URLs and files with support for PDF, HTML, TXT"""

    @staticmethod
    def load_from_url(url: str, cache_dir: str) -> Optional[Document]:
        """
        Load document from URL with caching
        """
        try:
            cache_path = Path(cache_dir) / "documents" / f"{hashlib.md5(url.encode()).hexdigest()}.txt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if cached
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata = {'source': url, 'type': 'url', 'content_type': 'cached', 'length': len(content)}
                logger.info(f"Loaded cached document from {url}")
                return Document(
                    id=hashlib.md5(url.encode()).hexdigest(),
                    content=content,
                    metadata=metadata
                )
            
            logger.info(f"Downloading from URL: {url}")
            response = requests.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            logger.info(f"Content-Type: {content_type}")
            
            if 'text/html' in content_type or 'text/plain' in content_type:
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
                content = ' '.join(soup.get_text().split())
            elif 'application/pdf' in content_type:
                pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
                content = ''
                if pdf_reader.is_encrypted:
                    try:
                        pdf_reader.decrypt('')
                    except Exception as e:
                        logger.error(f"Failed to decrypt PDF from {url}: {e}")
                        return None
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i}: {e}")
            else:
                content = response.text
            
            if not content or len(content.strip()) < 100:
                logger.error(f"Document too short or empty: {len(content)} chars")
                return None
            
            doc_id = hashlib.md5(url.encode()).hexdigest()
            
            # Cache the document
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f" Successfully loaded and cached document from {url}: {len(content)} characters")
            
            return Document(
                id=doc_id,
                content=content,
                metadata={
                    'source': url,
                    'type': 'url',
                    'content_type': content_type,
                    'length': len(content)
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load URL {url}: {e}", exc_info=True)
            return None

    @staticmethod
    def load_from_file(file_path: str, cache_dir: str) -> Optional[Document]:
        """
        Load document from file with caching and decryption handling
        """
        try:
            path = Path(file_path)
            cache_path = Path(cache_dir) / "documents" / f"{hashlib.md5(file_path.encode()).hexdigest()}.txt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if cached and file hasn't changed
            if cache_path.exists() and path.exists() and path.stat().st_mtime <= cache_path.stat().st_mtime:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                source_name = str(path) if len(str(path)) < 100 else path.name
                metadata = {
                    'source': source_name,
                    'source_filename': path.name,
                    'type': 'file',
                    'path': str(path),
                    'length': len(content)
                }
                logger.info(f"Loaded cached document from {file_path}")
                return Document(
                    id=hashlib.md5(file_path.encode()).hexdigest(),
                    content=content,
                    metadata=metadata
                )
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            logger.info(f"Loading file: {file_path}")
            
            suffix = path.suffix.lower()
            
            if suffix == '.pdf':
                with open(path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ''
                    if pdf_reader.is_encrypted:
                        try:
                            pdf_reader.decrypt('')
                        except Exception as e:
                            logger.error(f"Failed to decrypt PDF {file_path}: {e}")
                            return None
                    for page in pdf_reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n"
                        except Exception as e:
                            logger.warning(f"Failed to extract page: {e}")
            elif suffix in ['.txt', '.md', '.html', '.htm']:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if suffix in ['.html', '.htm']:
                    soup = BeautifulSoup(content, 'html.parser')
                    for script in soup(['script', 'style']):
                        script.decompose()
                    content = ' '.join(soup.get_text().split())
            else:
                logger.error(f"Unsupported file type: {suffix}")
                return None
            
            if not content or len(content.strip()) < 100:
                logger.error(f"Document too short: {len(content)} chars")
                return None
            
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
            
            # Cache the document
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f" Successfully loaded and cached file {path.name}: {len(content)} characters")

            # Use full path for source to be more descriptive
            source_name = str(path) if len(str(path)) < 100 else path.name

            return Document(
                id=doc_id,
                content=content,
                metadata={
                    'source': source_name,
                    'source_filename': path.name,
                    'type': 'file',
                    'path': str(path),
                    'length': len(content)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}", exc_info=True)
            return None

class EmbeddingGenerator:
    """Generate embeddings with caching"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device  # e.g., "cuda:0", "cuda:1", "cpu", or None for auto
        self.model = None
        self.cache = {}
        self._initialize()

    def _initialize(self):
        """Load embedding model with error handling"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            if self.device:
                logger.info(f" Setting embedding model device to: {self.device}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self.model = SentenceTransformer(self.model_name)
            logger.info(f" Embedding model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generate embeddings with in-memory caching and batch processing

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding (default 64 for optimal GPU utilization)
                       - CPU: 32 recommended
                       - GPU (8GB): 64-128 recommended
                       - GPU (16GB+): 256-512 recommended

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                results.append(self.cache[text_hash])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits > 0:
            hit_rate = (cache_hits / len(texts)) * 100
            logger.debug(f" Embedding cache: {cache_hits}/{len(texts)} hits ({hit_rate:.1f}% hit rate)")

        if uncached_texts:
            try:
                logger.info(f" Generating embeddings for {len(uncached_texts)} texts (batch_size={batch_size})...")
                embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )

                for text, embedding in zip(uncached_texts, embeddings):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self.cache[text_hash] = embedding.tolist()

                for i, idx in enumerate(uncached_indices):
                    results[idx] = embeddings[i].tolist()

                logger.info(f" Generated {len(uncached_texts)} embeddings successfully")

            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                dim = self.model.get_sentence_embedding_dimension()
                return [[0.0] * dim for _ in texts]

        return results

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

class DocumentChunker:
    """Chunk documents with sliding window"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Dict[str, Any]]:
        """Chunk document with sliding window"""
        chunks = []
        content = document.content
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end]
            
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    'document_id': document.id,
                    'chunk_id': f"{document.id}_{chunk_id}",
                    'text': chunk_text,
                    'metadata': {
                        'start': start,
                        'end': end,
                        'source': document.metadata.get('source', ''),
                        'chunk_size': self.chunk_size,
                        'overlap': self.overlap
                    }
                })
                chunk_id += 1
            
            start += self.chunk_size - self.overlap
        
        logger.info(f"Created {len(chunks)} chunks from document {document.metadata.get('source', 'Unknown')}")
        return chunks

class VectorStore:
    """Vector store using FAISS for cosine similarity search with persistent caching and GPU acceleration"""

    def __init__(self, embedding_generator: EmbeddingGenerator, cache_db_path: str, use_gpu: bool = False, gpu_device: int = 0):
        self.embedding_generator = embedding_generator
        self.dimension = embedding_generator.get_dimension()
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device  # Which GPU to use (0, 1, 2, etc.)
        self.gpu_available = False
        self.gpu_resources = None

        # Initialize CPU index first (safe default)
        self.index = faiss.IndexFlatIP(self.dimension)

        # GPU acceleration with opt-in to prevent CUDA architecture mismatch crashes
        # Set CLINORCHESTRA_ENABLE_GPU=1 environment variable to enable GPU
        # Note: Requires FAISS compiled for matching CUDA architecture
        enable_gpu_env = os.getenv('CLINORCHESTRA_ENABLE_GPU', '0').lower() in ('1', 'true', 'yes')
        auto_detect_gpu = use_gpu or enable_gpu_env

        if auto_detect_gpu:
            gpu_success = False
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(" GPU requested - Attempting FAISS GPU initialization...")
                    logger.info(" NOTE: This requires FAISS compiled for your CUDA architecture")

                    try:
                        # Create GPU resources
                        self.gpu_resources = faiss.StandardGpuResources()

                        # Test GPU compatibility with a comprehensive test
                        test_index = faiss.IndexFlatIP(self.dimension)
                        test_vectors = np.random.rand(10, self.dimension).astype('float32')
                        faiss.normalize_L2(test_vectors)
                        test_index.add(test_vectors)

                        # Try to move test index to GPU
                        logger.info(f" Moving test index to GPU {self.gpu_device}...")
                        gpu_test_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, test_index)

                        # Perform a comprehensive test search to verify GPU works
                        # This will trigger GPU kernel execution including L2Norm
                        logger.info(" Running GPU compatibility test...")
                        test_query = np.random.rand(1, self.dimension).astype('float32')
                        faiss.normalize_L2(test_query)
                        D, I = gpu_test_index.search(test_query, 1)

                        # Verify results are valid
                        if D is not None and I is not None and len(D[0]) > 0:
                            # GPU test successful! Now move the real index to GPU
                            del gpu_test_index
                            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, self.index)
                            self.gpu_available = True
                            gpu_success = True
                            mode_info = "(explicitly enabled)" if use_gpu else "(environment variable)"
                            logger.info(f" FAISS GPU mode: ACTIVE on GPU {self.gpu_device} {mode_info} - 10-90x faster searches!")
                        else:
                            logger.warning("GPU test returned invalid results")
                            raise RuntimeError("GPU test validation failed")

                    except Exception as gpu_error:
                        # GPU initialization or test failed - fall back to CPU
                        error_msg = str(gpu_error)
                        logger.warning(f"GPU FAISS initialization failed: {type(gpu_error).__name__}: {error_msg}")

                        # Provide helpful error message for common CUDA errors
                        if "209" in error_msg or "no kernel image" in error_msg.lower():
                            logger.error(" CUDA Error 209: FAISS was compiled for a different GPU architecture")
                            logger.error(" Your GPU is not compatible with the FAISS build.")
                            logger.error(" To fix: Reinstall FAISS compiled for your specific CUDA compute capability")

                        logger.info(" Falling back to CPU mode...")

                        # Clean up GPU resources if allocated
                        if self.gpu_resources is not None:
                            try:
                                del self.gpu_resources
                                self.gpu_resources = None
                            except:
                                pass

                        # Ensure we have a clean CPU index
                        self.index = faiss.IndexFlatIP(self.dimension)
                        self.gpu_available = False
                        logger.info(f" FAISS CPU mode: ACTIVE (GPU incompatible)")

                else:
                    logger.info(f" FAISS CPU mode: ACTIVE (CUDA not available via PyTorch)")

            except ImportError:
                logger.info(f" FAISS CPU mode: ACTIVE (PyTorch not installed)")
            except Exception as e:
                logger.warning(f"Unexpected error during GPU detection: {type(e).__name__}: {str(e)}")
                logger.info(f" FAISS CPU mode: ACTIVE (GPU detection error)")
                # Ensure CPU index is initialized
                if not gpu_success:
                    self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # GPU not requested - use CPU mode
            logger.info(f" FAISS CPU mode: ACTIVE (default)")
            logger.info(f" To enable GPU: Set environment variable CLINORCHESTRA_ENABLE_GPU=1")

        self.chunks = []
        self.documents = {}
        self.cache_db_path = cache_db_path

        # Initialize SQLite database schema
        self._initialize_db()

        logger.info(f"VectorStore initialized (dimension={self.dimension})")

    def _initialize_db(self):
        """Initialize SQLite database schema"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT PRIMARY KEY,
                        source TEXT,
                        content TEXT,
                        metadata TEXT,
                        hash TEXT
                    )
                """)
                # Create embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT,
                        chunk_text TEXT,
                        embedding BLOB,
                        chunk_config TEXT,
                        FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
                    )
                """)
                conn.commit()
                logger.info(f"Initialized SQLite database at {self.cache_db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    def add_documents(self, documents: List[Document], chunker: DocumentChunker, app_state):
        """Add documents to vector store: chunk, embed, index using cached data"""
        all_chunks = []
        new_documents = []
        
        # Load cached documents and embeddings
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            for doc in documents:
                cursor.execute("SELECT content, metadata, hash FROM documents WHERE doc_id = ?", (doc.id,))
                result = cursor.fetchone()
                if result:
                    cached_content, cached_metadata, cached_hash = result
                    current_hash = hashlib.md5(doc.content.encode()).hexdigest()
                    if current_hash == cached_hash:
                        doc.metadata = json.loads(cached_metadata)
                        self.documents[doc.id] = doc
                        chunks = chunker.chunk(doc)
                        all_chunks.extend(chunks)
                    else:
                        new_documents.append(doc)
                else:
                    new_documents.append(doc)
            
            # Load cached embeddings
            for chunk in all_chunks:
                cursor.execute("SELECT embedding, chunk_config FROM embeddings WHERE chunk_id = ?", (chunk['chunk_id'],))
                result = cursor.fetchone()
                if result:
                    embedding, chunk_config = result
                    config = json.loads(chunk_config)
                    if (config['chunk_size'] == chunker.chunk_size and
                        config['overlap'] == chunker.overlap):
                        chunk['embedding'] = pickle.loads(embedding)
        
        # Process new documents
        for doc in new_documents:
            self.documents[doc.id] = doc
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
            # Cache new document
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO documents (doc_id, source, content, metadata, hash) VALUES (?, ?, ?, ?, ?)",
                    (doc.id, doc.metadata.get('source', ''), doc.content, json.dumps(doc.metadata),
                     hashlib.md5(doc.content.encode()).hexdigest())
                )
                conn.commit()
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return
        
        logger.info(f"Total chunks to process: {len(all_chunks)}")
        
        # Generate embeddings for chunks without cached embeddings
        chunk_texts = []
        chunk_indices = []
        for i, chunk in enumerate(all_chunks):
            if 'embedding' not in chunk or chunk['embedding'] is None:
                chunk_texts.append(chunk['text'])
                chunk_indices.append(i)
        
        if chunk_texts:
            embeddings = self.embedding_generator.generate(chunk_texts)
            for i, embedding in zip(chunk_indices, embeddings):
                all_chunks[i]['embedding'] = embedding
                # Cache embedding
                with sqlite3.connect(self.cache_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO embeddings (chunk_id, doc_id, chunk_text, embedding, chunk_config) VALUES (?, ?, ?, ?, ?)",
                        (
                            all_chunks[i]['chunk_id'],
                            all_chunks[i]['document_id'],
                            all_chunks[i]['text'],
                            pickle.dumps(embedding),
                            json.dumps({
                                'chunk_size': chunker.chunk_size,
                                'overlap': chunker.overlap
                            })
                        )
                    )
                    conn.commit()
        
        # Add to FAISS index
        valid_embeddings = []
        valid_chunks = []
        
        for chunk in all_chunks:
            if 'embedding' in chunk and chunk['embedding'] and len(chunk['embedding']) == self.dimension:
                valid_embeddings.append(chunk['embedding'])
                valid_chunks.append(chunk)
        
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            self.chunks.extend(valid_chunks)
            
            logger.info(f" Added {len(valid_chunks)} chunks to vector store")
            logger.info(f"Total vectors in index: {self.index.ntotal}")

    def search(self, query: str, k: int = 3) -> List[SearchResult]:
        """Search for relevant chunks using cosine similarity"""
        if not self.chunks:
            logger.warning("No chunks in vector store")
            return []
        
        try:
            query_embeddings = self.embedding_generator.generate([query])
            if not query_embeddings or not query_embeddings[0]:
                return []
            
            query_embedding = np.array([query_embeddings[0]], dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            
            k = min(k, len(self.chunks))
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and score > 0.1:
                    chunk = self.chunks[idx]
                    doc_id = chunk['document_id']
                    
                    if doc_id in self.documents:
                        result = SearchResult(
                            document=self.documents[doc_id],
                            score=float(score),
                            chunk_text=chunk['text']
                        )
                        results.append(result)
            
            logger.info(f"Found {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

class RAGEngine:
    """
    RAG Engine with persistent caching
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.cache_dir = config.get('cache_dir', './rag_cache')
        self.use_gpu_faiss = config.get('use_gpu_faiss', False)
        self.gpu_device = config.get('gpu_device', 0)  # Which GPU to use (0, 1, 2, etc.)

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.embedding_generator = None
        self.chunker = None
        self.vector_store = None

        self.initialized = False
        self.documents_loaded = []

        # PERFORMANCE: Result caching for repeated queries
        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_bypass = False  # Can be set to True to force recompute

        logger.info("RAGEngine created (not initialized yet)")

    def initialize(self, sources: List[str], app_state) -> bool:
        """
        Initialize RAG engine with sources, using cached data when possible
        """
        try:
            if not sources:
                logger.warning("No sources provided to RAG engine")
                return False

            # Clear query cache when re-initializing with new documents
            self.query_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Cleared RAG query cache for re-initialization")
            
            logger.info("=" * 60)
            logger.info("INITIALIZING RAG ENGINE")
            logger.info("=" * 60)
            logger.info(f"Sources: {len(sources)}")
            logger.info(f"Embedding Model: {self.embedding_model}")
            logger.info(f"Chunk Size: {self.chunk_size}")
            logger.info(f"Chunk Overlap: {self.chunk_overlap}")
            logger.info("")
            
            # Initialize components
            logger.info("Step 1: Loading embedding model...")
            # Determine device for embedding model
            # If GPU is requested, use the specified GPU device (cuda:N)
            # Otherwise let SentenceTransformer auto-detect
            embedding_device = None
            if self.use_gpu_faiss or (hasattr(app_state, 'optimization_config') and
                                     hasattr(app_state.optimization_config, 'use_gpu_faiss') and
                                     app_state.optimization_config.use_gpu_faiss):
                try:
                    import torch
                    if torch.cuda.is_available():
                        embedding_device = f"cuda:{self.gpu_device}"
                        logger.info(f" Using GPU device {self.gpu_device} for embeddings")
                except ImportError:
                    logger.warning(" PyTorch not available - embedding model will use CPU")

            self.embedding_generator = EmbeddingGenerator(self.embedding_model, device=embedding_device)

            logger.info("Step 2: Initializing chunker...")
            self.chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)

            logger.info("Step 3: Initializing vector store...")
            # Get use_gpu_faiss from config or app_state
            use_gpu = self.use_gpu_faiss
            if hasattr(app_state, 'optimization_config') and hasattr(app_state.optimization_config, 'use_gpu_faiss'):
                use_gpu = app_state.optimization_config.use_gpu_faiss

            self.vector_store = VectorStore(self.embedding_generator, Path(self.cache_dir) / "rag_cache.db",
                                          use_gpu=use_gpu, gpu_device=self.gpu_device)
            
            # Load documents
            logger.info("Step 4: Loading documents...")
            documents = []
            
            for i, source in enumerate(sources, 1):
                logger.info(f"Loading source {i}/{len(sources)}: {source}")
                
                if source.startswith('http://') or source.startswith('https://'):
                    doc = DocumentLoader.load_from_url(source, self.cache_dir)
                else:
                    doc = DocumentLoader.load_from_file(source, self.cache_dir)
                
                if doc:
                    documents.append(doc)
                    self.documents_loaded.append(source)
                    logger.info(f" Loaded: {source}")
                else:
                    logger.warning(f" Failed to load: {source}")
            
            if not documents:
                logger.error("No documents successfully loaded")
                return False
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            # Add to vector store
            logger.info("Step 5: Chunking and embedding documents...")
            self.vector_store.add_documents(documents, self.chunker, app_state)
            
            if self.vector_store.index.ntotal > 0:
                self.initialized = True
                logger.info("")
                logger.info("=" * 60)
                logger.info(" RAG ENGINE INITIALIZED SUCCESSFULLY")
                logger.info("=" * 60)
                logger.info(f"Documents Loaded: {len(documents)}")
                logger.info(f"Total Chunks: {self.vector_store.index.ntotal}")
                logger.info(f"Embedding Dimension: {self.vector_store.dimension}")
                logger.info("=" * 60)
                return True
            else:
                logger.error("No vectors in index after initialization")
                return False
                
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}", exc_info=True)
            return False

    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """Retrieve relevant chunks for a query"""
        if not self.initialized:
            logger.warning("RAG engine not initialized")
            return []
        
        logger.info(f"RAG Query: '{query[:100]}...'")
        return self.vector_store.search(query, k)

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            'initialized': self.initialized,
            'embedding_model': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'num_documents': len(self.vector_store.documents) if self.vector_store else 0,
            'num_chunks': len(self.vector_store.chunks) if self.vector_store else 0,
            'dimension': self.vector_store.dimension if self.vector_store else 0,
            'documents_loaded': self.documents_loaded
        }

    def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Query RAG engine (alias for retrieve with dict output)
        Maintains backward compatibility

        PERFORMANCE: Results are cached based on query text and k value.
        """
        # PERFORMANCE: Create cache key from query and k
        cache_key = f"{query_text.strip().lower()}|k={k}"

        # Check cache first (unless bypass is enabled)
        if not self.cache_bypass and cache_key in self.query_cache:
            self.cache_hits += 1
            cached_result = self.query_cache[cache_key]
            logger.debug(f"Cache HIT for RAG query '{query_text[:50]}...' - returning {len(cached_result)} cached results")
            return cached_result

        self.cache_misses += 1

        results = self.retrieve(query_text, k)

        # Convert SearchResult objects to dictionaries
        dict_results = []
        for result in results:
            dict_results.append({
                'text': result.chunk_text,
                'score': result.score,
                'source': result.document.metadata.get('source', 'Unknown'),
                'metadata': result.document.metadata
            })

        # Cache the result
        self.query_cache[cache_key] = dict_results
        logger.debug(f"Cached RAG result for query '{query_text[:50]}...'")

        return dict_results

    def query_with_variations(self,
                              primary_query: str,
                              variations: List[str],
                              k: int = 10) -> List[Dict[str, Any]]:
        """
        Query RAG engine with term variations for improved recall and leniency.

        This method implements the search strategy by combining primary query with
        term variations (synonyms, abbreviations, related terms) to cast a wider
        net for relevant chunks.

        Args:
            primary_query: Main query string with core keywords
            variations: List of term variations (synonyms, abbreviations, related terms)
            k: Number of unique results to return

        Returns:
            List of deduplicated search results combining all term queries

        Example:
            primary_query = "ASPEN pediatric malnutrition diagnostic criteria"
            variations = ["undernutrition", "PEM", "SAM", "wasting", "stunting"]
            results = engine.query_with_variations(primary_query, variations, k=10)
        """
        if not variations:
            # No variations provided, use standard query
            return self.query(primary_query, k)

        # Build expanded query combining primary + variations
        variation_str = " ".join(variations)
        expanded_query = f"{primary_query} {variation_str}"

        logger.info(f"RAG query with variations: '{primary_query}' + {len(variations)} terms")
        logger.debug(f"  Variations: {variations[:5]}{'...' if len(variations) > 5 else ''}")

        # Fetch more results to account for potential duplicates
        fetch_k = min(k * 2, 50)
        results = self.query(expanded_query, k=fetch_k)

        # Deduplicate based on content similarity
        unique_results = []
        seen_content = set()

        for result in results:
            content = result.get('text', '').strip()
            if not content:
                continue

            # Use first 200 chars as uniqueness check (balance between accuracy and performance)
            content_signature = content[:200].lower()

            if content_signature not in seen_content:
                unique_results.append(result)
                seen_content.add(content_signature)

                if len(unique_results) >= k:
                    break

        logger.info(f"  Retrieved {len(unique_results)} unique results from {len(results)} total")

        return unique_results

    def clear_query_cache(self):
        """Clear the query result cache"""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("RAGEngine query cache cleared")

    def set_cache_bypass(self, bypass: bool):
        """Set cache bypass mode"""
        self.cache_bypass = bypass
        logger.info(f"RAGEngine cache bypass set to: {bypass}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.query_cache),
            'bypass_enabled': self.cache_bypass
        }
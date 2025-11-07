#!/usr/bin/env python3
"""
RAG Configuration Tab for ClinOrchestra - Document retrieval and embedding configuration
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: HeiderLab
Version: 1.0.0
"""

import gradio as gr
from typing import Dict, Any
import logging
from pathlib import Path
from core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

def create_rag_tab(app_state) -> Dict[str, Any]:
    """
    Create RAG configuration tab for document sources and embedding settings
    
    Args:
        app_state: Application state manager
        
    Returns:
        Dictionary of Gradio components
    """
    
    components = {}
    
    gr.Markdown("### RAG (Retrieval-Augmented Generation)")
    gr.Markdown("""
    Configure document sources and retrieval settings for evidence-based extraction refinement.
    
    **Note:** The RAG refinement prompt and query field selection are configured in the **Prompt Configuration** tab.
    This tab focuses on document sources, embeddings, and retrieval settings.
    """)
    
    rag_enabled = gr.Checkbox(
        label="Enable RAG",
        value=app_state.rag_config.enabled
    )
    components['rag_enabled'] = rag_enabled
    
    with gr.Column(visible=app_state.rag_config.enabled) as rag_config_panel:
        
        gr.Markdown("#### Document Sources")
        gr.Markdown("""
        Add documents that will be used to provide evidence for refining extractions.
        The system will automatically avoid re-downloading documents that are already cached.
        """)
        
        with gr.Tabs():
            with gr.Tab("URLs"):
                rag_urls = gr.Textbox(
                    label="Document URLs (one per line)",
                    placeholder="https://example.com/document.pdf\nhttps://example.com/guide.html",
                    lines=5,
                    value="\n".join([doc for doc in app_state.rag_config.documents if doc.startswith('http')])
                )
                components['rag_urls'] = rag_urls
            
            with gr.Tab("File Paths"):
                rag_paths = gr.Textbox(
                    label="File Paths (one per line)",
                    placeholder="/path/to/document1.pdf\n/path/to/document2.txt",
                    lines=5,
                    value="\n".join([doc for doc in app_state.rag_config.documents if not doc.startswith('http')])
                )
                components['rag_paths'] = rag_paths
            
            with gr.Tab("Upload"):
                rag_files = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".html"]
                )
                components['rag_files'] = rag_files
        
        gr.Markdown("---")
        gr.Markdown("#### Embedding Configuration")
        
        with gr.Row():
            rag_embedding_model = gr.Dropdown(
                choices=[
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ],
                value=app_state.rag_config.embedding_model,
                label="Embedding Model",
                info="Model used to create document embeddings"
            )
            components['rag_embedding_model'] = rag_embedding_model
        
        with gr.Row():
            rag_chunk_size = gr.Slider(
                minimum=128, maximum=1024, value=app_state.rag_config.chunk_size, step=64,
                label="Chunk Size",
                info="Size of text chunks for embedding (characters)"
            )
            components['rag_chunk_size'] = rag_chunk_size
            
            rag_chunk_overlap = gr.Slider(
                minimum=0, maximum=200, value=app_state.rag_config.chunk_overlap, step=10,
                label="Chunk Overlap",
                info="Overlap between consecutive chunks (characters)"
            )
            components['rag_chunk_overlap'] = rag_chunk_overlap
        
        gr.Markdown("---")
        gr.Markdown("#### Retrieval Configuration")
        
        rag_k_value = gr.Slider(
            minimum=1, maximum=10, value=app_state.rag_config.k_value, step=1,
            label="Chunks to Retrieve (k)",
            info="Number of most relevant chunks to retrieve for each query"
        )
        components['rag_k_value'] = rag_k_value
        
        gr.Markdown("---")
        gr.Markdown("#### Cache Management")
        gr.Markdown("""
        The RAG system caches embeddings to avoid reprocessing the same documents.
        Embeddings are only recreated when the embedding model or chunk configuration changes.
        """)
        
        with gr.Row():
            cache_info = gr.HTML(
                value="<div>Cache information will be displayed here after initialization</div>"
            )
            components['cache_info'] = cache_info
        
        with gr.Row():
            clear_cache_btn = gr.Button("Clear Cache", variant="secondary")
            components['clear_cache_btn'] = clear_cache_btn
        
        gr.Markdown("---")
        gr.Markdown("#### Initialize RAG")
        
        init_rag_btn = gr.Button("Initialize RAG System", variant="primary", size="lg")
        components['init_rag_btn'] = init_rag_btn
        
        rag_status = gr.TextArea(
            label="Status",
            lines=12,
            interactive=False
        )
        components['rag_status'] = rag_status
    
    components['rag_config_panel'] = rag_config_panel
    
    def toggle_rag_panel(enabled):
        """Toggle RAG panel visibility"""
        return gr.update(visible=enabled)
    
    def get_cache_info():
        """Get cache information"""
        try:
            cache_dir = app_state.rag_config.cache_dir
            cache_path = Path(cache_dir)
            
            if not cache_path.exists():
                return "<div>No cache directory found</div>"
            
            db_files = list(cache_path.glob("*.db"))
            embedding_files = list(cache_path.glob("*.pkl"))
            
            cache_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            cache_size_mb = cache_size / (1024 * 1024)
            
            info = f"""
            <div style="padding: 10px; background: #f0f0f0; border-radius: 5px;">
                <strong>Cache Status:</strong><br/>
                Directory: {cache_dir}<br/>
                Database files: {len(db_files)}<br/>
                Embedding files: {len(embedding_files)}<br/>
                Total size: {cache_size_mb:.2f} MB
            </div>
            """
            return info
            
        except Exception as e:
            return f"<div>Error getting cache info: {str(e)}</div>"
    
    def clear_cache():
        """Clear RAG cache"""
        try:
            cache_dir = Path(app_state.rag_config.cache_dir)
            if cache_dir.exists():
                for file in cache_dir.rglob("*"):
                    if file.is_file():
                        file.unlink()
                
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                return "Cache cleared successfully", get_cache_info()
            else:
                return "No cache directory found", get_cache_info()
                
        except Exception as e:
            return f"Error clearing cache: {str(e)}", get_cache_info()
                                      
    def initialize_rag(enabled, urls, paths, files, embedding_model,
                      chunk_size, chunk_overlap, k_value):
        """
        Initialize RAG system with all sources and intelligent caching
        
        Args:
            enabled: Whether RAG is enabled
            urls: URLs as newline-separated string
            paths: File paths as newline-separated string
            files: Uploaded files
            embedding_model: Embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k_value: Number of chunks to retrieve
            
        Returns:
            Status message string, cache info
        """
        if not enabled:
            return "RAG not enabled", get_cache_info()
        
        try:
            sources = []
            
            if urls and urls.strip():
                url_list = [u.strip() for u in urls.split('\n') if u.strip()]
                sources.extend(url_list)
                logger.info(f"Added {len(url_list)} URLs")
            
            if paths and paths.strip():
                path_list = [p.strip() for p in paths.split('\n') if p.strip()]
                sources.extend(path_list)
                logger.info(f"Added {len(path_list)} file paths")
            
            if files:
                if isinstance(files, list):
                    file_paths = [f.name for f in files]
                    sources.extend(file_paths)
                    logger.info(f"Added {len(file_paths)} uploaded files")
                else:
                    sources.append(files.name)
                    logger.info(f"Added 1 uploaded file")
            
            if not sources:
                return "Error: No document sources provided. Add URLs, file paths, or upload files.", get_cache_info()
            
            logger.info(f"Initializing RAG with {len(sources)} sources")
            
            config_changed = False
            current_config = {
                'embedding_model': embedding_model,
                'chunk_size': int(chunk_size),
                'chunk_overlap': int(chunk_overlap)
            }
            
            if (app_state.rag_config.embedding_model != embedding_model or
                app_state.rag_config.chunk_size != int(chunk_size) or
                app_state.rag_config.chunk_overlap != int(chunk_overlap)):
                config_changed = True
                logger.info("Embedding configuration changed - cache will be refreshed")
            
            rag_engine = RAGEngine({
                'embedding_model': embedding_model,
                'chunk_size': int(chunk_size),
                'chunk_overlap': int(chunk_overlap),
                'cache_dir': app_state.rag_config.cache_dir,
                'force_refresh': config_changed
            })
            
            success = rag_engine.initialize(sources, app_state)
            
            if not success:
                return "Error: Failed to initialize RAG engine. Check logs.", get_cache_info()
            
            success = app_state.set_rag_config(
                enabled=True,
                documents=sources,
                embedding_model=embedding_model,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                k_value=int(k_value),
                initialized=True
            )
            
            if not success:
                return "Error: RAG initialized but failed to save configuration", get_cache_info()
            
            app_state.set_rag_engine(rag_engine)
            
            stats = rag_engine.get_stats()
            
            query_fields = getattr(app_state.prompt_config, 'rag_query_fields', [])
            
            status = f"""RAG System Initialized Successfully
    
    **Documents:**
    - Total Sources: {len(sources)}
    - URLs: {len([s for s in sources if s.startswith('http')])}
    - Files: {len([s for s in sources if not s.startswith('http')])}
    - Total Chunks: {stats['num_chunks']}
    
    **Embedding:**
    - Model: {embedding_model}
    - Chunk Size: {chunk_size} chars
    - Overlap: {chunk_overlap} chars
    - Dimension: {stats['dimension']}
    
    **Retrieval Configuration:**
    - K Value: {k_value} chunks per query
    - Query Fields: {', '.join(query_fields) if query_fields else 'Will use extracted data fields'}
    
    **Caching:**
    - Configuration changed: {'Yes' if config_changed else 'No'}
    - Cache directory: {app_state.rag_config.cache_dir}
    
    **Status:** Ready for Processing
    
    **Note:** RAG refinement prompt and query fields can be configured in the Prompt Configuration tab."""
            
            logger.info("RAG initialization completed")
            return status, get_cache_info()
            
        except Exception as e:
            logger.error(f"RAG init failed: {e}", exc_info=True)
            return f"Error: {str(e)}\n\nCheck console logs for details.", get_cache_info()
            
    
    rag_enabled.change(
        fn=toggle_rag_panel,
        inputs=[rag_enabled],
        outputs=[rag_config_panel]
    )
    
    clear_cache_btn.click(
        fn=clear_cache,
        outputs=[rag_status, cache_info]
    )
    
    init_rag_btn.click(
        fn=initialize_rag,
        inputs=[
            rag_enabled, rag_urls, rag_paths, rag_files,
            rag_embedding_model, rag_chunk_size, rag_chunk_overlap,
            rag_k_value
        ],
        outputs=[rag_status, cache_info]
    )
    
    app_state.observer.subscribe(
        "tab_loaded",
        lambda: cache_info.update(value=get_cache_info())
    )
    
    return components
import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path so we can import raganything without pip install
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Mock/Import check
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
except ImportError as e:
    print(f"Error importing RAG-Anything: {e}")
    print("Please make sure you are running this script from the RAG-Anything repository root.")
    sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Run RAG-Anything with vLLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name served by vLLM")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM API Base URL")
    parser.add_argument("--api-key", default="EMPTY", help="vLLM API Key")
    parser.add_argument("--input-file", required=True, help="Path to input PDF/Document")
    parser.add_argument("--query", default="What is the summary of this document?", help="Query to ask")
    
    args = parser.parse_args()

    print(f"🚀 Starting RAG-Anything with vLLM Backend")
    print(f"   Model: {args.model}")
    print(f"   URL:   {args.api_base}")
    print(f"   Input: {args.input_file}")

    # Configuration for RAG-Anything
    # We disable vision to keep it simple and 'text/table' focused as requested
    config = RAGAnythingConfig(
        working_dir="./rag_storage_vllm",
        parser="mineru",          # Using MinerU as default
        parse_method="auto",      
        enable_image_processing=False, # Disable complex vision for stability
        enable_table_processing=True,
        enable_equation_processing=True
    )

    # LLM Function wrapping vLLM (OpenAI Compatible)
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            model=args.model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=args.api_key,
            base_url=args.api_base,
            **kwargs
        )

    # Embedding Function
    # NOTE: In a real vLLM setup, you usually run a separate embedding server or use a local lib.
    # Here we assume vLLM or another service at the same URL provides embeddings OR 
    # we fall back to a local one if the user prefers. 
    # For simplicity in this demo, we assume the endpoint supports embeddings or we use a standard OpenAI one.
    # IF vLLM doesn't serve embeddings, this will fail. 
    # ADAPTATION: Use a local HuggingFace embedding if possible, but let's try the endpoint first.
    
    # Actually, vLLM *can* serve embeddings, but often doesn't by default. 
    # Let's use a safe local fallback if possible, or assume the user has set it up.
    # For now, let's point to the SAME endpoint assuming the user served an embedding model or compatible.
    
    # To be safer for the demo, let's use a dummy embedding or a standard one if the user has a key.
    # But since we want "offline" / "vLLM", we should assume an embedding model is available.
    # Let's try to use the same endpoint.
    
    embedding_func = EmbeddingFunc(
        embedding_dim=768, # Adjust based on model
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=args.model, # Often vLLM uses the same model name for embeddings if supported
            api_key=args.api_key,
            base_url=args.api_base
        )
    )

    # Initialize RAG
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func
    )

    # Ingest Document
    print("\n[1] Ingesting Document...")
    await rag.process_document_complete(
        file_path=args.input_file,
        output_dir="./output_vllm",
        parse_method="auto"
    )

    # Query
    print(f"\n[2] Querying: '{args.query}'")
    result = await rag.aquery(args.query, mode="hybrid")
    
    print("\n" + "="*50)
    print("ANSWER FROM vLLM:")
    print(result)
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

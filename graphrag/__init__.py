__all__ = ["GraphRAGPipeline"]


def __getattr__(name: str):
    if name == "GraphRAGPipeline":
        from .pipeline.graphrag_pipeline import GraphRAGPipeline

        return GraphRAGPipeline
    raise AttributeError(f"module 'graphrag' has no attribute {name!r}")

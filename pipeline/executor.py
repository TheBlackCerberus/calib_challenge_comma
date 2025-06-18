from typing import Any, Union, Tuple
import asyncio
import time
import logging
from pipeline.base import SourceStage, PipelineStage, StageContext

logger = logging.getLogger(__name__)

class Pipeline:
    """Simple async pipeline executor"""
    
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.source: SourceStage | None = None
        self.stages: list[PipelineStage] = []
        
    def set_source(self, source: SourceStage) -> 'Pipeline':
        """Set the data source"""
        self.source = source
        return self
        
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """Add a processing stage"""
        self.stages.append(stage)
        return self
        
    async def run(self, ctx: StageContext) -> list[Any]:
        """Run the pipeline"""
        if not self.source:
            raise ValueError("No source stage set")
            
        results: list[Any] = []
        start_time = time.time()
        
        logger.info(f"Starting pipeline '{self.name}'")
        
        # Process streaming data
        async for item in self.source.generate(ctx):
            # Pass through each stage
            current = item
            
            for stage in self.stages:
                if current is None:
                    break
                    
                try:
                    current = await stage.process(current, ctx)
                    stage.processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error in stage {stage.name}: {e}")
                    current = None
                    
            if current is not None:
                results.append(current)
        
        # Finalize stages
        logger.info("Finalizing stages...")
        for stage in self.stages:
            try:
                final_result = await stage.finalize(ctx)
                if final_result is not None:
                    results.append(final_result)
            except Exception as e:
                logger.error(f"Error finalizing stage {stage.name}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f}s. Processed {len(results)} items.")
        
        return results
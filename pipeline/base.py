from abc import ABC, abstractmethod
from typing import Generic, TypeVar, AsyncIterator, Union, Tuple, Any
from dataclasses import dataclass
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')

@dataclass
class StageContext:
    """Context passed through pipeline stages"""
    video_path: str
    camera_matrix: np.ndarray
    total_frames: int | None = None

class PipelineStage(ABC, Generic[TInput, TOutput]):
    """Base class for pipeline stages"""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        
    @abstractmethod
    async def process(self, input_data: TInput, ctx: StageContext) -> Union[TOutput, Tuple[Any, ...], None]:
        """Process a single input and produce output(s)"""
        pass
    
    async def finalize(self, ctx: StageContext) -> Union[TOutput, Tuple[Any, ...], None]:
        """Called after all inputs processed"""
        return None

class SourceStage(ABC, Generic[TOutput]):
    """Base class for source stages that generate data"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    async def generate(self, ctx: StageContext) -> AsyncIterator[TOutput]:
        """Generate output items"""
        pass
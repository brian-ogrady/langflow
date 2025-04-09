from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langflow.base.compressors.model import LCCompressorComponent
from langflow.io import StrInput
from langflow.template.field.base import Output


class HuggingFaceRerankComponent(LCCompressorComponent):
    display_name = "HuggingFace Rerank"
    description = "Rerank documents using the HuggingFace Cross Encoders"
    name = "HuggingFaceRerank"
    icon = "HuggingFace"

    inputs = [
        *LCCompressorComponent.inputs,
        StrInput(
            name="model_name",
            display_name="Model NAme",
            value="BAAI/bge-reranker-base",
        ),
    ]

    outputs = [
        Output(
            display_name="Reranked Documents",
            name="reranked_documents",
            method="compress_documents",
        ),
    ]

    def build_compressor(self) -> CrossEncoderReranker:  # type: ignore[type-var]
        try:
            model = HuggingFaceCrossEncoder(model_name=self.model_name, model_kwargs={"device": "cpu"})
        except ImportError as e:
            msg = "Please check the model name and try again."
            raise ImportError(msg) from e
        return CrossEncoderReranker(
            model=model,
            top_n=self.top_n,
        )

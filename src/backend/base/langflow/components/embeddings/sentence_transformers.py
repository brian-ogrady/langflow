from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import StrInput


class SentenceTransformersEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "Sentence Transformers"
    description = "Generate embeddings using sentence-transformers"
    icon = "HuggingFace"
    name = "SentenceTransformersEmbeddings"

    inputs = [
        StrInput(
            name="model_name",
            display_name="Sentence Transformers Model",
            value="sentence-transformers/all-mpnet-base-v2",
            required=True,
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        try:
            output = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={"device": "cpu"})
        except Exception as e:
            msg = (
                "Unable to create the Sentence Transformers embeddings. ",
                "Please verify the model name, ensure the relevant model is pulled, and try again.",
            )
            raise ValueError(msg) from e
        return output

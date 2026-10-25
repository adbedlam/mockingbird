from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")


def build_local_expression(annotated_conversation_path: Path):
    annotated_conversation = _load_conversation(annotated_conversation_path)
    topics: list[str] = sorted(annotated_conversation["topic"].unique())
    
    # load the csv, turn into DF
    # extract unique topics and embed them using the model
    # add the embeddings to the DF
    # pass the df to _train_model
    return  # the trained model

def _load_conversation(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _extract_topics(annotated_conversation: pd.DataFrame) -> list[str]:
    return sorted(annotated_conversation["topic"].dropna().unique().tolist())

def _embed_topics(topics: list[str]) -> dict[str, list[float]]:
    embeddings = model.encode(
        [f"query: {topic}" for topic in topics],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return {
        topic: embedding.tolist()
        for topic, embedding in zip(topics, embeddings)
    }

def _add_topic_embeddings(
    annotated_conversation: pd.DataFrame,
    topic_embeddings: dict[str, list[float]],
) -> pd.DataFrame:
    annotated_conversation = annotated_conversation.copy()
    annotated_conversation["topic_embedding"] = (
        annotated_conversation["topic"].map(topic_embeddings)
    )
    return annotated_conversation

def _train_model(annotated_conversation: pd.DataFrame):  # -> trained model type
    raise NotImplementedError("Training the local expression model is not yet implemented.")

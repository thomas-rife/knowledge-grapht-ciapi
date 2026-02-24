import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Tuple

from knowledge_graph import CausalKnowledgeGraph

app = FastAPI()

class ReviewRequest(BaseModel):
    nodes: List[str]
    edges: List[Tuple[str, str]]
    class_topic_progressions: List[Dict[str, float]]
    student_topic_progressions: Dict[str, float]

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/review")
def generate_review(request: ReviewRequest):
    try:
        kg = CausalKnowledgeGraph(
            nodes=request.nodes,
            edges=request.edges,
            class_topic_progressions=request.class_topic_progressions,
            student_topic_progressions=request.student_topic_progressions,
        )

        review = kg.determine_review_topics(
            student_topic_progressions=request.student_topic_progressions
        )

        # Ensure all values are plain Python floats for JSON serialization
        review = {k: float(v) for k, v in review.items()}

        return review

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
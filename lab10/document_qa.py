import re
from qa_model import ExtractiveQAModel


class DocumentQASystem:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", max_length=384, stride=128):
        self.qa_engine = ExtractiveQAModel(
            model_name=model_name,
            max_length=max_length,
            stride=stride
        )

    def split_document(self, document: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n+", document) if p.strip()]
        return paragraphs if paragraphs else [document.strip()]

    def answer_question(self, question: str, document: str, top_k: int = 3) -> dict:
        paragraphs = self.split_document(document)
        candidates = []

        for idx, para in enumerate(paragraphs):
            res = self.qa_engine.predict(question=question, context=para)
            if res["answer"]:
                candidates.append({
                    "paragraph_index": idx,
                    "answer": res["answer"],
                    "score": res["score"],
                    "char_span": res["char_span"],
                    "source_paragraph": para
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0] if candidates else {"answer": "", "score": 0.0, "paragraph_index": -1}

        return {
            "question": question,
            "best_answer": best["answer"],
            "confidence_score": best["score"],
            "paragraph_index": best["paragraph_index"],
            "top_candidates": candidates[:top_k]
        }


if __name__ == "__main__":
    doc = (
        "Extractive Question Answering locates the exact substring within a given context that answers a question.\n\n"
        "Unlike generative models, extractive QA models predict start and end token probabilities over the input sequence.\n\n"
        "Evaluation typically relies on Exact Match (EM) and token-level F1 score."
    )
    system = DocumentQASystem()
    result = system.answer_question("What metrics are typically used for evaluation?", doc)
    print("Best Answer:", result["best_answer"])
    print("Score:", result["confidence_score"])

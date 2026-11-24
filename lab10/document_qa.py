import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class DocumentQASystem:
    """
    Mini Extractive Document QA System:
    User question + Document -> Sliding Window Tokenization -> QA Model -> Best Answer Span
    """
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", max_length=384, stride=128):
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def split_paragraphs(self, document: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n+", document) if p.strip()]
        return paragraphs if paragraphs else [document.strip()]

    def _predict_on_chunk(self, question: str, context: str, n_best: int = 10, max_ans_len: int = 35) -> dict:
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        offsets_all = inputs.pop("offset_mapping").tolist()
        sample_map = inputs.pop("overflow_to_sample_mapping").tolist()
        seq_ids_all = [inputs.sequence_ids(i) for i in range(len(sample_map))]

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()

        best_score = -1e9
        best_answer = ""
        best_span = (0, 0)

        for i in range(len(sample_map)):
            s_logit = start_logits[i]
            e_logit = end_logits[i]
            offsets = offsets_all[i]
            seq_ids = seq_ids_all[i]

            for s in np.argsort(s_logit)[-n_best:][::-1]:
                for e in np.argsort(e_logit)[-n_best:][::-1]:
                    if seq_ids[s] != 1 or seq_ids[e] != 1:
                        continue
                    if e < s or (e - s + 1) > max_ans_len:
                        continue
                    score = float(s_logit[s] + e_logit[e])
                    if score > best_score:
                        best_score = score
                        c_start, c_end = offsets[s][0], offsets[e][1]
                        best_answer = context[c_start:c_end]
                        best_span = (c_start, c_end)

        return {"answer": best_answer, "score": round(best_score, 4), "char_span": best_span}

    def ask(self, question: str, document: str) -> dict:
        paragraphs = self.split_paragraphs(document)
        best_result = {"answer": "", "score": -1e9, "paragraph_index": -1, "char_span": (0, 0)}

        for idx, para in enumerate(paragraphs):
            res = self._predict_on_chunk(question, para)
            if res["answer"] and res["score"] > best_result["score"]:
                best_result = {
                    "answer": res["answer"],
                    "score": res["score"],
                    "paragraph_index": idx,
                    "char_span": res["char_span"]
                }

        return {
            "question": question,
            "answer": best_result["answer"],
            "confidence_logit_sum": best_result["score"],
            "paragraph_index": best_result["paragraph_index"],
            "char_span": best_result["char_span"]
        }


if __name__ == "__main__":
    # Tài liệu mẫu (Wikipedia-style về kiến trúc Transformer)
    wiki_document = (
        "The Transformer is a deep learning architecture introduced in 2017 by researchers at Google Brain "
        "in the landmark paper 'Attention Is All You Need'. Unlike recurrent neural networks (RNNs), "
        "Transformers process entire sequences in parallel using self-attention mechanisms.\n\n"
        "BERT (Bidirectional Encoder Representations from Transformers) was developed by Google in 2018. "
        "It uses only the encoder stack of the Transformer and is pre-trained using Masked Language Modeling "
        "and Next Sentence Prediction on BooksCorpus and English Wikipedia.\n\n"
        "DistilBERT is a distilled version of BERT introduced by Hugging Face in 2019. It retains 97% of "
        "BERT's language understanding capabilities while being 40% smaller and 60% faster during inference."
    )

    sample_questions = [
        "When was the Transformer architecture introduced?",
        "What paper introduced the Transformer?",
        "What tasks were used to pre-train BERT?",
        "How much faster is DistilBERT compared to BERT?"
    ]

    qa_app = DocumentQASystem()
    print("=== MINI DOCUMENT QA SYSTEM DEMO ===\n")
    for q in sample_questions:
        out = qa_app.ask(q, wiki_document)
        print(f"Q: {out['question']}")
        print(f"A: {out['answer']} (Paragraph #{out['paragraph_index']}, Score: {out['confidence_logit_sum']})\n")

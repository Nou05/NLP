import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class ExtractiveQAModel:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", max_length=384, stride=128):
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, question: str, context: str, n_best_size: int = 10, max_answer_len: int = 30) -> dict:
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

        offset_mappings = inputs.pop("offset_mapping").tolist()
        sample_mapping = inputs.pop("overflow_to_sample_mapping").tolist()
        seq_ids_list = [inputs.sequence_ids(i) for i in range(len(sample_mapping))]

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()

        best_score = -1e9
        best_answer = ""
        best_span = (0, 0)

        for i in range(len(sample_mapping)):
            s_logit = start_logits[i]
            e_logit = end_logits[i]
            offsets = offset_mappings[i]
            seq_ids = seq_ids_list[i]

            start_indexes = np.argsort(s_logit)[-n_best_size:][::-1]
            end_indexes = np.argsort(e_logit)[-n_best_size:][::-1]

            for s in start_indexes:
                for e in end_indexes:
                    if seq_ids[s] != 1 or seq_ids[e] != 1:
                        continue
                    if e < s or (e - s + 1) > max_answer_len:
                        continue
                    score = float(s_logit[s] + e_logit[e])
                    if score > best_score:
                        best_score = score
                        char_start, char_end = offsets[s][0], offsets[e][1]
                        best_answer = context[char_start:char_end]
                        best_span = (char_start, char_end)

        return {
            "answer": best_answer,
            "score": round(best_score, 4),
            "char_span": best_span
        }


if __name__ == "__main__":
    qa = ExtractiveQAModel()
    sample_ctx = "DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base."
    sample_q = "How was DistilBERT trained?"
    print(qa.predict(sample_q, sample_ctx))

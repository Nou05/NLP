# Lưu tại: lab09/metrics_utils.py
def get_entities(seq):
    """Trích xuất danh sách (entity_type, start_idx, end_idx) từ chuỗi nhãn BIO."""
    entities = []
    entity_type, start = None, None
    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            if entity_type is not None:
                entities.append((entity_type, start, i - 1))
            entity_type = tag[2:]
            start = i
        elif tag.startswith("I-"):
            if entity_type is None or tag[2:] != entity_type:
                if entity_type is not None:
                    entities.append((entity_type, start, i - 1))
                entity_type = tag[2:]
                start = i
        else:  # 'O'
            if entity_type is not None:
                entities.append((entity_type, start, i - 1))
                entity_type, start = None, None
    if entity_type is not None:
        entities.append((entity_type, start, len(seq) - 1))
    return entities

def compute_entity_metrics(true_labels, true_predictions):
    """Tính Precision, Recall, F1 ở mức thực thể (Entity-level)."""
    tp = fp = fn = 0
    for y_true, y_pred in zip(true_labels, true_predictions):
        true_ents = set(get_entities(y_true))
        pred_ents = set(get_entities(y_pred))
        tp += len(true_ents & pred_ents)
        fp += len(pred_ents - true_ents)
        fn += len(true_ents - pred_ents)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
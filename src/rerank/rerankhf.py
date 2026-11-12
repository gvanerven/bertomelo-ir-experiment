from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

template = "Query: {query}\nSentence: {document}"

def rank(query, documents, tokenizer, model, device, max_length=2048, normalize_scores=True):
    model.to(device)
    texts = [template.format(query=query, document=document) for document in documents]

    inputs = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, batch_size=64)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

        # Get the predicted classes and confidence scores
        predicted_classes = torch.argmax(probabilities, dim=1).tolist()
        confidences = probabilities.max(dim=1).values.tolist()

    # Construct the results
    results = [
        {"prediction": pred, "confidence": conf}
        for pred, conf in zip(predicted_classes, confidences)
    ]

    final_results = []
    for document, result in zip(documents, results):
        # If the prediction is 0, then get the score as 1 - confidence
        if result['prediction'] == 0:
            result['confidence'] = 1 - result['confidence']
        final_results.append((document, result['confidence']))
    
    # Sort by the confidence score, descending
    # sorted_results = sorted(final_results, key=lambda x: x[1], reverse=True)

    if normalize_scores:
        total_score = sum([result[1] for result in final_results])
        #total_score = sum([result[1] for result in sorted_results])
        #sorted_results = [(result[0], result[1] / total_score) for result in sorted_results]
        final_results = [(result[0], result[1] / total_score) for result in final_results]

    return final_results

def main():
    model_id = "cnmoro/Tucano160m-Reranker-EnPt"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    query = "Qual a receita de Fuji?"
    documents = [
        "A receita de Fuji é simples e deliciosa.",
        "Fuji é uma variedade de maçã muito popular.",
        "Para fazer a receita de Fuji, você precisará de ingredientes frescos.",
        "A receita de Fuji é conhecida por seu sabor único.",
    ]

    ranked_results = rank(query, documents, tokenizer, model, device, normalize_scores=True)
    for doc, score in ranked_results:
        print(f"Document: {doc}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
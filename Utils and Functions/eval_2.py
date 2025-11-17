import torch
import numpy as np

from metrics import recall_at_k, mrr, ndcg


@torch.inference_mode()
def evaluate_retrieval(translated_embd, image_embd, gt_indices, max_indices = 99, batch_size=100):
    """Evaluate retrieval performance using cosine similarity
    Args:
        translated_embd: (N_captions, D) translated caption embeddings
        image_embd: (N_images, D) image embeddings
        gt_indices: (N_captions,) ground truth image indices for each caption
        max_indices: number of top predictions to consider
    Returns:
        results: dict of evaluation metrics
    
    """
    # Compute similarity matrix
    if isinstance(translated_embd, np.ndarray):
        translated_embd = torch.from_numpy(translated_embd).float()
    if isinstance(image_embd, np.ndarray):
        image_embd = torch.from_numpy(image_embd).float()
    
    n_queries = translated_embd.shape[0]
    device = translated_embd.device
    
    # Prepare containers for the fragments to be reassembled
    all_sorted_indices = []
    l2_distances = []
    
    # Process in batches - the narrow gate approach
    for start_idx in range(0, n_queries, batch_size):
        batch_slice = slice(start_idx, min(start_idx + batch_size, n_queries))
        batch_translated = translated_embd[batch_slice]
        batch_img_embd = image_embd[batch_slice]
        
        # Compute similarity only for this batch
        batch_similarity = batch_translated @ batch_img_embd.T

        # Get top-k predictions for this batch
        batch_indices = batch_similarity.topk(k=max_indices, dim=1, sorted=True).indices.numpy()
        all_sorted_indices.append(gt_indices[batch_slice][batch_indices])

        # Compute L2 distance for this batch
        batch_gt = gt_indices[batch_slice]
        batch_gt_embeddings = image_embd[batch_gt]
        batch_l2 = (batch_translated - batch_gt_embeddings).norm(dim=1)
        l2_distances.append(batch_l2)
    
    # Reassemble the fragments
    sorted_indices = np.concatenate(all_sorted_indices, axis=0)
    
    # Apply the sacred metrics to the whole
    metrics = {
        'mrr': mrr,
        'ndcg': ndcg,
        'recall_at_1': lambda preds, gt: recall_at_k(preds, gt, 1),
        'recall_at_3': lambda preds, gt: recall_at_k(preds, gt, 3),
        'recall_at_5': lambda preds, gt: recall_at_k(preds, gt, 5),
        'recall_at_10': lambda preds, gt: recall_at_k(preds, gt, 10),
        'recall_at_50': lambda preds, gt: recall_at_k(preds, gt, 50),
    }
    
    results = {
        name: func(sorted_indices, gt_indices)
        for name, func in metrics.items()
    }
    
    l2_dist = torch.cat(l2_distances, dim=0).mean().item()
    results['l2_dist'] = l2_dist
    
    return results


def evaluate_model_on_validation(model, val_dataset, device):
    """Evaluate model on validation set"""
    model.eval()

    # Generate predictions
    val_predictions = []
    val_image_embeddings = []

    print(f"Generating predictions for {len(val_dataset):,} validation samples...")

    with torch.no_grad():
        for i in range(len(val_dataset)):
            text_emb, target_img_emb = val_dataset[i]
            text_emb = text_emb.unsqueeze(0).to(device)
            
            # Predict image embedding
            pred_img_emb = model(text_emb)
            val_predictions.append(pred_img_emb.cpu())
            
            # Store ground truth image embedding
            val_image_embeddings.append(target_img_emb)

    # Stack all embeddings
    translated_embd = torch.cat(val_predictions, dim=0).float()
    image_embd = torch.stack(val_image_embeddings).float()
    gt_indices = np.arange(len(val_dataset))

    print(f"   Searching among {image_embd.shape[0]:,} validation images")

    # Evaluate retrieval performance
    results = evaluate_retrieval(
        translated_embd=translated_embd,
        image_embd=image_embd,
        gt_indices=gt_indices,
        max_indices=99,
        batch_size=100
    )

    print(f"🎯 Results:")
    print(f"   MRR:        {results['mrr']:.4f}")
    print(f"   Recall@1:   {results['recall_at_1']:.4f}")
    print(f"   Recall@5:   {results['recall_at_5']:.4f}")
    print(f"   Recall@10:  {results['recall_at_10']:.4f}")
    print(f"   NDCG:       {results['ndcg']:.4f}")
    print(f"   L2 dist:    {results['l2_dist']:.4f}")

    return results
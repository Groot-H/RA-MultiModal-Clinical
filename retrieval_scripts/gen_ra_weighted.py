import os
import argparse
import numpy as np
import faiss
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Weighted (LDA-Raw) RA Splits")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original split folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save RA splits")
    parser.add_argument("--k", type=int, default=3, help="Number of items to retrieve")
    return parser.parse_args()

def load_data(path):
    lines_content = []
    paths = []
    labels = []
    attrs = []
    
    if not os.path.exists(path):
        return [], [], [], np.array([])

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            lines_content.append(line)
            parts = line.split('\t')
            paths.append(parts[0])
            labels.append(int(parts[1]))
            attrs.append([float(x) for x in parts[2:]])
            
    return lines_content, paths, np.array(labels), np.array(attrs)

def get_lda_weights_from_raw(X_raw, y):
    """Compute LDA weights using raw data"""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_raw, y)
    weights = np.abs(lda.coef_).mean(axis=0)
    weights /= weights.sum() # Normalize weights to sum to 1
    return weights.astype(np.float32)

def write_ra_file(output_path, query_lines, retrieved_indices, db_lines, k, is_train_on_train):
    with open(output_path, 'w') as f:
        for i, query_line in enumerate(query_lines):
            out_str = query_line
            
            # When train searches train, skip the first hit (itself)
            start_idx = 1 if is_train_on_train else 0
            end_idx = start_idx + k
            
            current_indices = retrieved_indices[i][start_idx:end_idx]
            
            for idx in current_indices:
                out_str += "\t" + db_lines[idx]
            
            f.write(out_str + "\n")

def process_fold(fold_idx, args):
    train_file = os.path.join(args.input_dir, f"train_fold{fold_idx}.txt")
    test_file = os.path.join(args.input_dir, f"test_fold{fold_idx}.txt")
    
    # 1. Load data
    train_lines, train_paths, train_y, train_raw = load_data(train_file)
    test_lines, test_paths, test_y, test_raw = load_data(test_file)
    
    if len(train_lines) == 0:
        print(f"Skipping Fold {fold_idx}: No training data.")
        return

    # 2. Compute weights (based on raw data)
    # Note: strictly follow the requirement to compute LDA using raw data
    weights = get_lda_weights_from_raw(train_raw, train_y)
    # print(f"Fold {fold_idx} Weights: {weights}") 

    # 3. Min-Max normalization
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_raw).astype(np.float32)
    test_norm = scaler.transform(test_raw).astype(np.float32) if len(test_raw) > 0 else np.array([])

    # 4. Apply weights
    train_feats = train_norm * weights
    test_feats = test_norm * weights if len(test_norm) > 0 else np.array([])

    # 5. Build Faiss index (L2 normalized inner product)
    d = train_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # Backup and L2 normalize
    db_feats = train_feats.copy()
    faiss.normalize_L2(db_feats)
    index.add(db_feats)
    
    # === Process Train Set (self-retrieval) ===
    # Retrieve k+1 items so we can skip self later
    D_train, I_train = index.search(db_feats, args.k + 1)
    
    out_train_path = os.path.join(args.output_dir, f"train_fold{fold_idx}.txt")
    write_ra_file(out_train_path, train_lines, I_train, train_lines, args.k, is_train_on_train=True)
    
    # === Process Test Set ===
    if len(test_lines) > 0:
        query_feats = test_feats.copy()
        faiss.normalize_L2(query_feats)
        D_test, I_test = index.search(query_feats, args.k)
        
        out_test_path = os.path.join(args.output_dir, f"test_fold{fold_idx}.txt")
        write_ra_file(out_test_path, test_lines, I_test, train_lines, args.k, is_train_on_train=False)

    print(f"Fold {fold_idx} processed (Weighted).")

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    for fold in range(5):
        process_fold(fold, args)

if __name__ == "__main__":
    main()

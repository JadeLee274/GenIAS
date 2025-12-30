python tsne.py \
    --data MSL \
    --sub-dataset C-1 \
    --window-size 10 \
    --time-or-dir dir \
    --load-positive-augmentor exp/ckpt/ \
    --load-perturbator exp/ckpt/ \
    --non-constant-dim-tau 0.6 \
    --constant-dim-tau 0.6 \
    --anomaly-or-normal normal \


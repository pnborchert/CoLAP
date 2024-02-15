# Fine-tune XNLI in English
python train.py \
    --task "xnli" \
    --target_lang "en" \
    --plm "xlm-roberta-base" \
    --epochs 5 \
    --fp16

# Few-shot adaptation to Vietnamese with 50 labeled examples
python train.py \
    --task "xnli" \
    --source_lang "en" \
    --target_lang "vi" \
    --plm "xlm-roberta-base" \
    --K 50 \
    --epochs 10 \
    --load_ckpt "checkpoints/xnli-xlm-roberta-base-en-colap" \
    --eval_zs \
    --eval_fs \
    --fp16
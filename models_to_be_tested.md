```bash
python train.py \
  --combo_id tf_efficientnet_b4_progressive \
  --display_name "EfficientNet-B4 + Progressive" \
  --model_name tf_efficientnet_b4.ns_jft_in1k \
  --classifier_type progressive \
  --weights_filename tf_efficientnet_b4_progressive_best.pth \
  --class_names_filename tf_efficientnet_b4_class_names.txt \
  --epochs 100 \
  --nfolds 5 \
  --batch_size 32 \
  --lr 0.001
```
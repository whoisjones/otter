# Run

```python
accelerate launch train_bi_compressed.py -config configs/bi_encoder.json
accelerate launch train_bi_contrastive.py -config configs/bi_encoder.json
accelerate launch train_cross_compressed.py -config configs/cross_encoder.json
accelerate launch train_cross_compressed.py -config configs/cross_encoder.json
```
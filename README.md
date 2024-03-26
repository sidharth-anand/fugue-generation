# Data Preprocessing

- Download the dataset from the Baroque-NMT [repo](https://gitlab.com/skalo/baroque-nmt/-/tree/master?ref_type=heads)
- Run `add_multipart_tracks.py`
- Then run `encode.py` and `transpose.py`

# Training and Inference
- Run `train.py` with arguments to customize the model architecture.
- This should generate model checkpoints (in the `models` directory by default)
- Run `inference.py` and provide a task and the path to the trained checkpoint

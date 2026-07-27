# composer-classification-cnn-lstm
Our project aims to develop deep learning models that can automatically identify the composer of a piece of classical music using MIDI data. We are focusing on compositions by Johann Sebastian Bach, Ludwig van Beethoven, Frédéric Chopin, and Wolfgang Amadeus Mozart.

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the notebook for exploration:

Open `notebooks/composer_classification.ipynb` in Jupyter and run the cells.

3. Train the model:

```bash
python src/train.py --data-dir data --epochs 10 --batch-size 8 --time-steps 500 --out-dir models
```

4. Evaluate a trained model:

```bash
python src/evaluate.py --data-dir data --model models/final_model.h5 --time-steps 500 --out-dir models
```

Notes

- Composer labels are inferred from the MIDI parent folder when available. If you have a specific mapping, replace `infer_label_from_path` logic in `src/data_generator.py`.
- `pretty_midi` is required to parse MIDI files. On Windows, installation may require a C compiler for some optional dependencies.

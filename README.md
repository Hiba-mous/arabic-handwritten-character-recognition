# Arabic Handwritten Character Recognition

A Streamlit application for recognizing individual handwritten Arabic letters using a saved TensorFlow/Keras model. Draw a character on a canvas or upload an image to view the predicted class and the five highest model scores.

## Features

- Recognition across 28 Arabic letter classes.
- Freehand drawing with adjustable stroke width and colors.
- PNG and JPEG image upload.
- Top-five prediction chart.
- Cached model loading in the `ProApp.py` interface.

## How inference works

The application resizes the input to 32 × 32 pixels, converts it to grayscale, flips it horizontally, applies a Gaussian blur, rotates it 90 degrees counterclockwise, and normalizes pixel values to the range 0–1. It passes a tensor of shape `(1, 32, 32, 1)` to `best.keras`.

The displayed confidence is a model output score; it is not a measured accuracy or a guarantee that the prediction is correct. The transformation sequence is taken from the existing application and still needs verification against the original training pipeline.

## Local setup

Place the trained model `best.keras` alongside `ProApp.py` before running the app. The source is published; the model file still needs to be uploaded by the repository owner.

Required Python packages, as identified from imports:

```text
streamlit
streamlit-drawable-canvas
tensorflow
numpy
opencv-python
matplotlib
Pillow
```

Install the dependencies, then start the interface from its directory:

```sh
python -m pip install -r requirements.txt
python -m streamlit run ProApp.py
```

Dependency versions have not yet been validated. The original interface uses `st.experimental_rerun()` and older image-display arguments, which may require compatibility changes for the installed Streamlit version.

## Reproducibility and project status

The local project contains a saved model and several application variants. The training implementation, evaluation results, and verified dataset-to-label mapping have not yet been recovered. No accuracy claim is made here. This application has not yet been run as part of the portfolio preparation.

## Attribution

This was a joint (binôme) project by **Hiba Moussadek** and **Drissi Marwane**. The existing interface also credits supervision by **Prof. N Azouagh**, at Hassan II University / FST Mohammedia, academic year 2024–2025. Both collaborators are credited in the application.


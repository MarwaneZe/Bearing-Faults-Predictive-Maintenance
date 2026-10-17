"""Deep learning approach: raw-signal 1D-CNN + feature MLP.

Keras 3 with the JAX backend (TensorFlow ships no wheels for this Python; the
Keras API is identical). The backend env var MUST be set before `import keras`,
so it is set here: importing anything under `src.deep` configures it.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
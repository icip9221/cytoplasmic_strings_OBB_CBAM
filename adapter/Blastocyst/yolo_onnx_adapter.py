from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


class YoloONNXAdapter:
    """Thin ONNX Runtime wrapper for raw deployment and debugging inference."""

    def __init__(
        self,
        model: str | Path,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        model_path = Path(model)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers or ort.get_available_providers(),
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def predict(self, inputs: np.ndarray | dict[str, Any]) -> dict[str, np.ndarray]:
        feed = {self.input_names[0]: inputs} if isinstance(inputs, np.ndarray) else inputs
        outputs = self.session.run(self.output_names, feed)
        return dict(zip(self.output_names, outputs, strict=True))

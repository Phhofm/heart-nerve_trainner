"""fp32 → fp16 ONNX conversion (runs in the isolated venv, never in the frozen app).

    python _fp16.py in.onnx out.onnx
"""

from __future__ import annotations

import sys


def main() -> int:
    import onnx
    from onnxconverter_common import float16

    src, dst = sys.argv[1], sys.argv[2]
    model16 = float16.convert_float_to_float16(onnx.load(src))
    onnx.save_model(model16, dst, save_as_external_data=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
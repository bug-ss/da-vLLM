"""Install the DA attention patch at interpreter start.

Put this file's directory on ``PYTHONPATH`` and use the spawn start method, and
every process Python starts -- including vLLM's separately spawned attention
workers under tensor parallel > 1, which never construct the logits processor --
installs the patch before importing anything else (guide 6.1).

Configuration comes from ``DA_VLLM_CONFIG`` (a JSON object matching
:class:`da_vllm.config.DAConfig`).  Env-gated behavior is read exactly once,
here, at install time: a replayed CUDA graph will not re-read an environment
variable.
"""

import json
import os
import sys

_RAW = os.environ.get("DA_VLLM_CONFIG")
if _RAW:
    try:
        from da_vllm.config import DAConfig
        from da_vllm.masking.patch import install_patch

        install_patch(DAConfig.from_dict(json.loads(_RAW)))
    except Exception as exc:  # never take the interpreter down
        print(f"da: sitecustomize failed to install the patch: {exc}", file=sys.stderr)

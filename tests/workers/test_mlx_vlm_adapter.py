"""Pure tests for the mlx_vlm worker adapter — no ``mlx_vlm`` binding
required, since ``build_server_argv`` never imports it at module scope.
"""

from __future__ import annotations

import inspect

from sovereign.workers.mlx_vlm_adapter import build_server_argv


def test_build_server_argv_maps_kwargs_to_kebab_flags():
    argv = build_server_argv(
        {"max_tokens": 1024, "vision_cache_size": 8, "kv_bits": 4},
        model_path="/models/foo",
        draft_model_path=None,
        host="127.0.0.1",
        port=9000,
    )
    assert argv[:6] == ["--model", "/models/foo", "--host", "127.0.0.1", "--port", "9000"]
    assert argv[argv.index("--max-tokens") + 1] == "1024"
    assert argv[argv.index("--vision-cache-size") + 1] == "8"
    assert argv[argv.index("--kv-bits") + 1] == "4"
    assert "--draft-model" not in argv


def test_build_server_argv_maps_mtp_draft():
    argv = build_server_argv(
        {"draft_kind": "mtp", "draft_block_size": 4},
        model_path="/m",
        draft_model_path="/draft-mtp",
        host="h",
        port=1,
    )
    assert argv[argv.index("--draft-model") + 1] == "/draft-mtp"
    assert argv[argv.index("--draft-kind") + 1] == "mtp"
    assert argv[argv.index("--draft-block-size") + 1] == "4"


def test_build_server_argv_kv_flags_kebab_cased():
    argv = build_server_argv(
        {"kv_quant_scheme": "turboquant", "max_kv_size": 131072, "quantized_kv_start": 512},
        model_path="/m",
        draft_model_path=None,
        host="h",
        port=1,
    )
    assert argv[argv.index("--kv-quant-scheme") + 1] == "turboquant"
    assert argv[argv.index("--max-kv-size") + 1] == "131072"
    assert argv[argv.index("--quantized-kv-start") + 1] == "512"


def test_build_server_argv_bools_become_bare_flags():
    argv = build_server_argv(
        {"trust_remote_code": True, "enable_thinking": False},
        model_path="/m",
        draft_model_path=None,
        host="h",
        port=1,
    )
    assert "--trust-remote-code" in argv
    assert "--enable-thinking" not in argv


def test_build_server_argv_passthrough_escape_hatch():
    argv = build_server_argv(
        {"top_logprobs_k": 5}, model_path="/m", draft_model_path=None, host="h", port=1
    )
    assert argv[argv.index("--top-logprobs-k") + 1] == "5"


def test_build_server_argv_has_no_api_key_path():
    # The key travels via the MLX_VLM_SERVER_API_KEY environment variable the
    # server reads natively (see MlxVlmManager.start_env) — the builder must
    # not even offer a parameter that could put it on a ps-visible argv.
    assert "api_key" not in inspect.signature(build_server_argv).parameters

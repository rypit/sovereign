"""Native inference-engine services.

The engines here (``llama_cpp``, ``mlx_lm``) share the subprocess + HTTP-health +
psutil-metrics lifecycle in :mod:`sovereign.services.inference.base`
(``NativeEngineManager``); each only supplies config parsing, argv generation,
and engine-specific pre-flight checks. Like every service they self-register via
:func:`sovereign.core.registry.register_service`; ``sovereign.services`` walks
this package recursively, so a new engine is just another folder here.
"""

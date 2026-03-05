# tests/test_llm_config.py

from llm_local.llm_config import LocalLLMConfig


def test_default_config_values():
    """LocalLLMConfig should expose the documented default values."""
    cfg = LocalLLMConfig()

    assert cfg.model == "llama3.2:3b"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.timeout_seconds == 360
    assert cfg.default_temperature == 0.2
    assert cfg.default_max_tokens is None
    assert cfg.default_options == {}


def test_config_override_values():
    """Explicit constructor values should override defaults."""
    cfg = LocalLLMConfig(
        model="mistral",
        base_url="http://example.com:1234",
        timeout_seconds=10,
        default_temperature=0.7,
        default_max_tokens=256,
        default_options={"top_p": 0.9},
    )

    assert cfg.model == "mistral"
    assert cfg.base_url == "http://example.com:1234"
    assert cfg.timeout_seconds == 10
    assert cfg.default_temperature == 0.7
    assert cfg.default_max_tokens == 256
    assert cfg.default_options == {"top_p": 0.9}


def test_default_options_isolated_between_instances():
    """
    default_options should not be shared between instances
    (dataclass default_factory behavior).
    """
    cfg1 = LocalLLMConfig()
    cfg2 = LocalLLMConfig()

    cfg1.default_options["foo"] = "bar"

    assert cfg2.default_options == {}

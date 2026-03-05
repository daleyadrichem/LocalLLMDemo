# tests/test_request_builder.py

from llm_local.llm_config import LocalLLMConfig
from llm_local.ollama_request_builder import OllamaRequestBuilder


def make_builder():
    cfg = LocalLLMConfig(
        model="test-model",
        default_temperature=0.5,
        default_max_tokens=128,
        default_options={"top_p": 0.9},
    )
    return OllamaRequestBuilder(cfg)


def test_merged_options_with_defaults():
    builder = make_builder()

    opts = builder._merged_options(
        temperature=None,
        max_tokens=None,
        options=None,
    )

    assert opts["temperature"] == 0.5
    assert opts["num_predict"] == 128
    assert opts["top_p"] == 0.9


def test_merged_options_overrides_config():
    builder = make_builder()

    opts = builder._merged_options(
        temperature=0.7,
        max_tokens=256,
        options={"top_k": 50},
    )

    assert opts["temperature"] == 0.7
    assert opts["num_predict"] == 256
    assert opts["top_p"] == 0.9
    assert opts["top_k"] == 50


def test_merged_options_skip_num_predict_if_invalid():
    cfg = LocalLLMConfig(
        default_temperature=0.2,
        default_max_tokens=None,
        default_options={},
    )
    builder = OllamaRequestBuilder(cfg)

    opts = builder._merged_options(
        temperature=None,
        max_tokens=0,
        options=None,
    )

    assert "num_predict" not in opts
    assert opts["temperature"] == 0.2


def test_chat_payload_structure():
    builder = make_builder()

    payload = builder.chat_payload(
        model="model-x",
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        max_tokens=None,
        options=None,
        stream=False,
    )

    assert payload["model"] == "model-x"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["stream"] is False
    assert "options" in payload


def test_generate_payload_structure():
    builder = make_builder()

    payload = builder.generate_payload(
        model="model-x",
        prompt="hello",
        system_prompt="be helpful",
        temperature=0.3,
        max_tokens=50,
        options=None,
        stream=True,
    )

    assert payload["model"] == "model-x"
    assert payload["prompt"] == "hello"
    assert payload["system"] == "be helpful"
    assert payload["stream"] is True
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["num_predict"] == 50


def test_generate_payload_without_system_prompt():
    builder = make_builder()

    payload = builder.generate_payload(
        model="model-x",
        prompt="hello",
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        options=None,
        stream=False,
    )

    assert "system" not in payload

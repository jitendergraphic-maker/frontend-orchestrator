from app.adapters.cohere import CohereAdapter


def test_cohere_adapter_mocked_complete(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {
                "message": {"content": [{"type": "text", "text": "cohere ok"}]},
                "usage": {
                    "billed_units": {"input_tokens": 4, "output_tokens": 2},
                    "tokens": {"input_tokens": 10, "output_tokens": 2},
                },
            }

    request = {}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(
        "app.adapters.cohere.get_settings",
        lambda: type(
            "Settings",
            (),
            {"cohere_api_key": "cohere_test", "completion_max_tokens": 1536},
        )(),
    )
    monkeypatch.setattr("app.adapters.cohere.httpx.post", post)

    result = CohereAdapter().complete(
        api_model="command-r7b-12-2024",
        prompt="Say hi",
        timeout=10,
    )

    assert result.output == "cohere ok"
    assert result.tokens_in == 4
    assert result.tokens_out == 2
    assert request["url"] == "https://api.cohere.com/v2/chat"
    assert request["headers"]["Authorization"] == "Bearer cohere_test"
    assert request["json"]["model"] == "command-r7b-12-2024"

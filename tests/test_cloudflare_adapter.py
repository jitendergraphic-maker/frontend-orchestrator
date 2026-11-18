from app.adapters.cloudflare import CloudflareAdapter


def test_cloudflare_adapter_mocked_complete(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {
                "success": True,
                "result": {
                    "response": "cf ok",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            }

    request = {}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(
        "app.adapters.cloudflare.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "cloudflare_api_token": "cfut_test",
                "cloudflare_account_id": "acct123",
                "completion_max_tokens": 1536,
            },
        )(),
    )
    monkeypatch.setattr("app.adapters.cloudflare.httpx.post", post)

    result = CloudflareAdapter().complete(
        api_model="@cf/meta/llama-3.1-8b-instruct-fp8",
        prompt="Say hi",
        timeout=10,
    )

    assert result.output == "cf ok"
    assert result.tokens_in == 5
    assert result.tokens_out == 2
    assert "acct123/ai/run/" in request["url"]
    assert "@cf/meta/llama-3.1-8b-instruct-fp8" in request["url"]
    assert request["headers"]["Authorization"] == "Bearer cfut_test"
    assert request["json"]["messages"][0]["content"] == "Say hi"
    assert request["json"]["max_tokens"] == 1536

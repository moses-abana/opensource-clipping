import json
from urllib.parse import parse_qs, urlparse

from clipping.studio import broll


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        return self.body


def test_download_pixabay_broll_uses_api_key_and_vertical_variant(monkeypatch, tmp_path):
    api_payload = {
        "hits": [
            {
                "id": 17,
                "videos": {
                    "large": {"url": "https://cdn.test/landscape.mp4", "width": 1920, "height": 1080},
                    "medium": {"url": "https://cdn.test/portrait.mp4", "width": 720, "height": 1280},
                },
            }
        ]
    }
    requests = []

    def fake_urlopen(request):
        requests.append(request)
        if len(requests) == 1:
            return _Response(json.dumps(api_payload).encode())
        return _Response(b"synthetic video")

    monkeypatch.setattr(broll.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(broll.random, "choice", lambda candidates: candidates[0])
    monkeypatch.setattr(broll, "USED_PIXABAY_IDS", set())
    destination = tmp_path / "broll.mp4"

    assert broll.download_pixabay_broll("city lights", "9:16", str(destination), "test key")
    assert destination.read_bytes() == b"synthetic video"
    assert parse_qs(urlparse(requests[0].full_url).query) == {
        "key": ["test key"], "q": ["city lights"], "per_page": ["30"], "safesearch": ["true"]
    }
    assert requests[1].full_url == "https://cdn.test/portrait.mp4"


def test_download_pixabay_broll_skips_without_key(capsys, tmp_path):
    assert not broll.download_pixabay_broll("nature", "16:9", str(tmp_path / "unused.mp4"), "")
    assert "PIXABAY_API_KEY" in capsys.readouterr().out


def test_cli_config_prefers_pixabay_key_and_accepts_legacy_variable(monkeypatch, tmp_path):
    from clipping import config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXABAY_API_KEY", "new-key")
    monkeypatch.setenv("PEXELS_API_KEY", "legacy-name-key")
    assert config.build_config(["--url", "https://example.test/video"]).pixabay_api_key == "new-key"

    monkeypatch.setenv("PIXABAY_API_KEY", "")
    assert config.build_config(["--url", "https://example.test/video"]).pixabay_api_key == "legacy-name-key"

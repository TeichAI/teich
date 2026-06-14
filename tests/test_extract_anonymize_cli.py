import json
import sqlite3
from pathlib import Path
import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from teich.converter import convert_traces_to_training_data
from teich.cli import app

runner = CliRunner()
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_cli_output(output: str) -> str:
    return ANSI_ESCAPE_RE.sub("", output)


def test_extract_subcommands_exist():
    for provider in ("claude", "codex", "hermes", "pi"):
        result = runner.invoke(app, ["extract", provider, "--help"])

        assert result.exit_code == 0
        output = _plain_cli_output(result.output)
        assert "--output" in output
        assert "--out" in output
        assert "--sessions-dir" in output
        assert "--model" in output


def test_extract_codex_from_explicit_sessions_dir(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "session"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(output_dir)],
        input="n\n",
    )

    assert result.exit_code == 0
    extracted = output_dir / "session.jsonl"
    assert extracted.exists()
    assert (output_dir / "README.md").exists()
    assert "Extracted 1 codex trace" in result.output
    assert "Automatically scrambled 0 API keys, 0 email addresses, and 0 username references" in result.output
    assert "Data was written to" in result.output


def test_extract_hermes_from_explicit_state_db(tmp_path: Path):
    state_db = tmp_path / "state.db"
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at TEXT, model TEXT)")
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)"
        )
        connection.execute(
            "INSERT INTO sessions (id, source, started_at, model) VALUES (?, ?, ?, ?)",
            ("session/1", "cli", "2026-06-13T00:00:00Z", "test-model"),
        )
        connection.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("session/1", "user", "hello", "2026-06-13T00:00:01Z"),
        )
        connection.commit()
    finally:
        connection.close()

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        ["extract", "hermes", "--sessions-dir", str(state_db), "--output", str(output_dir)],
        input="n\n",
    )

    assert result.exit_code == 0
    extracted = output_dir / "hermes-agent-session-1.jsonl"
    assert extracted.exists()
    rows = [json.loads(line) for line in extracted.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "external_session_meta"
    assert rows[0]["payload"]["source"] == "hermes-agent"
    assert rows[1]["type"] == "external_message"
    assert rows[1]["role"] == "user"
    assert (output_dir / "README.md").exists()


def test_extract_defaults_to_data_folder_and_anonymizes_before_prompt(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    original_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    (sessions_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "test-model"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"hello from /home/alice/project alice@example.com {original_key}",
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["extract", "codex", "--sessions-dir", str(sessions_dir)], input="n\n")
        data_dir = Path("data")
        text = (data_dir / "session.jsonl").read_text(encoding="utf-8")

    assert result.exit_code == 0
    assert "Extracted 1 codex trace" in result.output
    assert "Automatically scrambled 1 API keys, 1 email addresses, and 1 username references" in result.output
    assert "Data was written to data" in result.output
    assert "Would you like to upload to Hugging Face?" in result.output
    assert "/home/user1/project" in text
    assert "user1@example.com" in text
    assert "sk-or-v1-" in text
    assert original_key not in text
    assert "alice@example.com" not in text


def test_extract_model_filter_for_codex_claude_pi_and_hermes(tmp_path: Path):
    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    pi_dir = tmp_path / "pi"
    codex_dir.mkdir()
    claude_dir.mkdir()
    pi_dir.mkdir()
    (codex_dir / "fable.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-fable", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    (codex_dir / "opus.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-opus", "model": "codex-opus"}}) + "\n",
        encoding="utf-8",
    )
    (claude_dir / "fable.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-fable-5", "content": []}})
        + "\n",
        encoding="utf-8",
    )
    (claude_dir / "opus.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-4-8", "content": []}})
        + "\n",
        encoding="utf-8",
    )
    (pi_dir / "fable.jsonl").write_text(
        json.dumps({"type": "session_info", "modelId": "pi-fable-5"}) + "\n",
        encoding="utf-8",
    )
    (pi_dir / "opus.jsonl").write_text(
        json.dumps({"type": "session_info", "modelId": "pi-opus"}) + "\n",
        encoding="utf-8",
    )
    state_db = tmp_path / "state.db"
    connection = sqlite3.connect(state_db)
    try:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at TEXT, model TEXT)")
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)"
        )
        for session_id, model in (("hermes-fable", "hermes-fable-5"), ("hermes-opus", "hermes-opus")):
            connection.execute(
                "INSERT INTO sessions (id, source, started_at, model) VALUES (?, ?, ?, ?)",
                (session_id, "cli", "2026-06-13T00:00:00Z", model),
            )
            connection.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "user", "hello", "2026-06-13T00:00:01Z"),
            )
        connection.commit()
    finally:
        connection.close()

    cases = [
        ("codex", codex_dir, "fable.jsonl"),
        ("claude", claude_dir, "fable.jsonl"),
        ("pi", pi_dir, "fable.jsonl"),
        ("hermes", state_db, "hermes-agent-hermes-fable.jsonl"),
    ]
    for provider, source, expected_file in cases:
        output_dir = tmp_path / f"out-{provider}"
        result = runner.invoke(
            app,
            [
                "extract",
                provider,
                "--sessions-dir",
                str(source),
                "--output",
                str(output_dir),
                "--model",
                "fable-5",
            ],
            input="n\n",
        )

        assert result.exit_code == 0, result.output
        assert f"Extracted 1 {provider} trace with fable-5" in result.output
        assert (output_dir / expected_file).exists()
        assert len(list(output_dir.glob("*.jsonl"))) == 1


def test_extract_claude_preserves_raw_order_and_only_anonymizes_inline(tmp_path: Path):
    sessions_dir = tmp_path / "claude"
    sessions_dir.mkdir()
    source_file = sessions_dir / "session.jsonl"
    secret_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    raw_lines = [
        (
            '{"type":"last-prompt","sessionId":"claude-session",'
            f'"lastPrompt":"Email alice@example.com key {secret_key} path /home/alice/project"}}\n'
        ),
        '{ "type": "mode", "mode": "normal", "sessionId": "claude-session" }\n',
        '{"type":"permission-mode","permissionMode":"bypassPermissions","sessionId":"claude-session"}\n',
        '{"type":"file-history-snapshot","sessionId":"claude-session","leafUuid":"leaf-1"}\n',
        (
            '{"type":"user","timestamp":"2026-06-10T00:00:01.000Z","sessionId":"claude-session",'
            '"uuid":"user-1","message":{"role":"user","content":"Build app"}}\n'
        ),
        (
            '{"type":"assistant","timestamp":"2026-06-10T00:00:02.000Z","sessionId":"claude-session",'
            '"parentUuid":"user-1","message":{"id":"msg-1","role":"assistant","model":"claude-fable-5",'
            '"content":[{"type":"text","text":"I will build it."}]}}\n'
        ),
        '{"type":"mode","mode":"normal","sessionId":"claude-session"}\n',
        '{"type":"permission-mode","permissionMode":"auto","sessionId":"claude-session"}\n',
    ]
    source_file.write_text("".join(raw_lines), encoding="utf-8")
    source_examples = convert_traces_to_training_data(source_file)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "extract",
            "claude",
            "--sessions-dir",
            str(sessions_dir),
            "--output",
            str(output_dir),
            "--model",
            "claude-fable-5",
        ],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Automatically scrambled 1 API keys, 1 email addresses, and 1 username references" in result.output
    extracted_file = output_dir / "session.jsonl"
    extracted_lines = extracted_file.read_text(encoding="utf-8").splitlines(keepends=True)
    assert [json.loads(line)["type"] for line in extracted_lines] == [
        "last-prompt",
        "mode",
        "permission-mode",
        "file-history-snapshot",
        "user",
        "assistant",
        "mode",
        "permission-mode",
    ]
    assert extracted_lines[1:] == raw_lines[1:]
    assert "alice@example.com" not in extracted_lines[0]
    assert secret_key not in extracted_lines[0]
    assert "/home/alice" not in extracted_lines[0]
    assert "user1@example.com" in extracted_lines[0]
    assert "sk-or-v1-" in extracted_lines[0]
    assert "/home/user1/project" in extracted_lines[0]
    assert convert_traces_to_training_data(extracted_file) == source_examples


def test_extract_refreshes_flat_output_before_writing(tmp_path: Path):
    sessions_dir = tmp_path / "codex"
    sessions_dir.mkdir()
    (sessions_dir / "fable.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-fable", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True)
    stale_file = output_dir / "old-opus.jsonl"
    stale_file.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-opus", "model": "codex-opus"}}) + "\n",
        encoding="utf-8",
    )
    stale_legacy_file = output_dir / "codex" / "old-nested-opus.jsonl"
    stale_legacy_file.parent.mkdir(parents=True)
    stale_legacy_file.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-nested-opus", "model": "codex-opus"}}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text("old staged path: /home/stale/project\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "extract",
            "codex",
            "--sessions-dir",
            str(sessions_dir),
            "--output",
            str(output_dir),
            "--model",
            "fable-5",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Automatically scrambled 0 API keys, 0 email addresses, and 0 username references" in result.output
    assert not stale_file.exists()
    assert not stale_legacy_file.exists()
    assert not stale_legacy_file.parent.exists()
    assert (output_dir / "fable.jsonl").exists()
    assert len(list(output_dir.glob("*.jsonl"))) == 1


def test_extract_can_upload_staged_anonymized_output_to_huggingface(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data"

    with patch("teich.cli.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/fable-traces"
        mock_api_cls.return_value = mock_api

        result = runner.invoke(
            app,
            [
                "extract",
                "codex",
                "--sessions-dir",
                str(sessions_dir),
                "--output",
                str(output_dir),
                "--model",
                "fable-5",
            ],
            input="y\narmand0e/fable-traces\n",
            env={"HF_TOKEN": "hf-test-token"},
        )

    assert result.exit_code == 0
    assert "Published dataset: https://huggingface.co/datasets/armand0e/fable-traces" in result.output
    mock_api_cls.assert_called_once_with(token="hf-test-token")
    mock_api.create_repo.assert_called_once_with(
        repo_id="armand0e/fable-traces",
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    mock_api.upload_large_folder.assert_called_once_with(
        repo_id="armand0e/fable-traces",
        folder_path=str(output_dir),
        repo_type="dataset",
        private=False,
        ignore_patterns=["partials/**", "failures/**"],
    )
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "armand0e/fable-traces" in readme
    assert 'path: "*.jsonl"' in readme


def test_extract_prompts_for_hf_token_when_env_token_is_missing(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "data"

    with patch("teich.cli.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/datasets/armand0e/fable-traces"
        mock_api_cls.return_value = mock_api

        result = runner.invoke(
            app,
            [
                "extract",
                "codex",
                "--sessions-dir",
                str(sessions_dir),
                "--output",
                str(output_dir),
                "--model",
                "fable-5",
            ],
            input="y\narmand0e/fable-traces\nhf-prompted-token\n",
            env={"HF_TOKEN": "", "HUGGINGFACE_HUB_TOKEN": "", "TEICH_HF_TOKEN": ""},
        )

    assert result.exit_code == 0
    assert "HF_TOKEN" in result.output
    mock_api_cls.assert_called_once_with(token="hf-prompted-token")


def test_extract_rejects_blank_huggingface_upload_inputs(tmp_path: Path):
    sessions_dir = tmp_path / "codex" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "session.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "session", "model": "codex-fable-5"}}) + "\n",
        encoding="utf-8",
    )

    with patch("teich.cli.HfApi") as mock_api_cls:
        blank_repo = runner.invoke(
            app,
            ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(tmp_path / "blank-repo")],
            input="y\n   \n",
            env={"HF_TOKEN": "hf-test-token"},
        )

    assert blank_repo.exit_code == 1
    assert "Hugging Face dataset repo id is required" in blank_repo.output
    mock_api_cls.assert_not_called()

    with patch("teich.cli.HfApi") as mock_api_cls:
        blank_token = runner.invoke(
            app,
            ["extract", "codex", "--sessions-dir", str(sessions_dir), "--output", str(tmp_path / "blank-token")],
            input="y\narmand0e/fable-traces\n   \n",
            env={"HF_TOKEN": "", "HUGGINGFACE_HUB_TOKEN": "", "TEICH_HF_TOKEN": ""},
        )

    assert blank_token.exit_code == 1
    assert "HF_TOKEN is required" in blank_token.output
    mock_api_cls.assert_not_called()


def test_anonymize_replaces_emails_keys_and_home_usernames_consistently(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    original_key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "path": "/home/alice/project",
                "windows_path": "C:\\Users\\alice\\Documents\\repo",
                "encoded_project_path": "projects/-home-alice-Documents-repo/session.jsonl\n-home-alice",
                "listing": "drwxrwxr-x 2 alice alice 4096 Jun 11 18:40 -home-alice",
                "email": "alice@example.com",
                "message": f"email alice@example.com key {original_key}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "alice" not in text
    assert "example.com" in text
    assert "user1@example.com" in text
    assert text.count("user1@example.com") == 2
    assert "/home/user1/project" in text
    assert "C:\\\\Users\\\\user1\\\\Documents" in text
    assert "projects/-home-user1-Documents-repo/session.jsonl" in text
    assert "-home-user1" in text
    assert "user1 user1 4096" in text
    assert "sk-or-v1-" in text
    assert original_key not in text
    assert "email=2" in result.output
    assert "username=7" in result.output
    assert "api_key=1" in result.output


def test_anonymize_generalizes_to_synthetic_secret_and_reference_matrix(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    raw_values = {
        "email": "dev.team+alerts@company.io",
        "sk_ant": "sk-ant-api03-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
        "sk_proj": "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
        "sk": "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK123456",
        "hf": "hf_abcdefghijklmnopqrstuvwxyz123456",
        "github_pat": "github_pat_abcdefghijklmnopqrstuvwxyz_1234567890",
        "gho": "gho_abcdefghijklmnopqrstuvwxyz123456",
        "glpat": "glpat-abcdefghijklmnopqrstuvwxyz123456",
        "xoxb": "xoxb-abcdefghijklmnopqrstuvwxyz-123456",
        "google": "AIzaabcdefghijklmnopqrstuvwxyz123456",
        "jwt": (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkphbmUifQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "bearer": "Bearer abcdefghijklmnopqrstuvwxyz1234567890",
        "env_secret": "PROJECT_SECRET=abcdefghijklmnopqrstuvwxyz1234567890",
    }
    image_data = "aGVsbG8=" * 40
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": "\n".join(
                    [
                        f"Contact {raw_values['email']} and then {raw_values['email']} again.",
                        "/home/zoe/app /Users/zoe/app D:\\Users\\zoe\\Documents\\app",
                        "projects/-Users-zoe-Documents-app/session.jsonl",
                        "drwxr-xr-x 2 zoe zoe 4096 Jun 13 09:00 app",
                        "Public path stays /Users/Public/Documents/shared",
                        " ".join(raw_values[key] for key in raw_values if key != "email"),
                        "Read @src/main.ts and @docs/*.md before editing.",
                        "Install @scope/package and keep @media, @keyframes, @pytest.mark, @app.get.",
                        "Remote git@gitlab.com:example/repo.git and https://medium.com/@someone/post stay.",
                        "Keep code expression process.env.PROJECT_SECRET intact.",
                    ]
                ),
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    for value in raw_values.values():
        assert value not in text
    assert "zoe" not in text
    assert text.count("@example.com") == 2
    assert "/home/user" in text
    assert "/Users/user" in text
    assert "D:\\\\Users\\\\user" in text
    assert "-Users-user" in text
    assert "/Users/Public/Documents/shared" in text
    assert "sk-ant-api03-" in text
    assert "sk-proj-" in text
    assert "sk-" in text
    assert "hf_" in text
    assert "github_pat_" in text
    assert "gho_" in text
    assert "glpat-" in text
    assert "xoxb-" in text
    assert "AIza" in text
    assert "eyJ0eXAiOiJKV1Qi." in text
    assert "Bearer redacted_" in text
    assert "PROJECT_SECRET=redacted_" in text
    assert "process.env.PROJECT_SECRET" in text
    assert image_data in text
    assert "redacted_base64" not in text
    assert "@src/main.ts" in text
    assert "@docs/*.md" in text
    assert "@scope/package" in text
    assert "@media" in text
    assert "@keyframes" in text
    assert "@pytest.mark" in text
    assert "@app.get" in text
    assert "git@gitlab.com:example/repo.git" in text
    assert "https://medium.com/@someone/post" in text


def test_anonymize_keeps_secret_replacements_consistent_per_trace(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    key = "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEFGHIJK"
    email = "operator@example.net"
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "first": f"{email} {key} /home/operator/app",
                "second": f"{email} {key} /Users/operator/app",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    emails = re.findall(r"user\d+@example\.com", text)
    keys = re.findall(r"sk-proj-[A-Za-z0-9]{32}", text)
    paths = re.findall(r"/(?:home|Users)/(user\d+)/app", text)
    assert len(set(emails)) == 1
    assert len(set(keys)) == 1
    assert len(set(paths)) == 1
    assert key not in text
    assert email not in text
    assert "operator" not in text


def test_anonymize_scrubs_env_style_keys_without_scrubbing_tokenizer_terms(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": "\n".join(
                    [
                        'CONTEXT7_API_KEY: "ctx7sk-ba627c37-3cea-45c2-8808-720374cce2a3"',
                        'HF_TOKEN="abcdefghijklmnopqrstuvwxyz1234567890"',
                        "export const GROQ_API_KEY =",
                        '  process.env.GROQ_API_KEY ?? "gsk_TESTabcdefghijklmnop";',
                        'tokenizer = "AutoTokenizer.from_pretrained"',
                        'PROCEDURAL_WARMUP_TOKENS = "abcdefghijklmnopqrstuvwxyz1234567890"',
                        'adapter.supportsLocalAgentJwt = "abcdefghijklmnopqrstuvwxyz1234567890"',
                    ]
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "ctx7sk-ba627c37-3cea-45c2-8808-720374cce2a3" not in text
    assert "abcdefghijklmnopqrstuvwxyz1234567890" in text
    assert "HF_TOKEN=\\\"redacted_" in text
    assert "ctx7sk-" in text
    assert "process.env.GROQ_API_KEY" in text
    assert "gsk_TESTabcdefghijklmnop" not in text
    assert "gsk_" in text
    assert "AutoTokenizer.from_pretrained" in text
    assert "PROCEDURAL_WARMUP_TOKENS" in text
    assert "adapter.supportsLocalAgentJwt" in text
    assert "api_key=3" in result.output


def test_anonymize_preserves_base64_media_payloads_without_touching_metadata(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    image_data = "aGVsbG8=" * 40
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    row = json.loads((output_dir / "trace.jsonl").read_text(encoding="utf-8"))
    source = row["content"][0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert source["data"] == image_data
    assert image_data in json.dumps(row)
    assert "image_data" not in result.output


def test_anonymize_does_not_treat_systemd_units_as_emails(tmp_path: Path):
    input_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "content": "\n".join(
                    [
                        "Failed to start org.gnome.Shell@x11.service",
                        "Read `@AGENT.md` before editing",
                        "Also inspect @README.md and -@plan.md",
                        "Read https://medium.com/@ahmed.soliman/why-anthropics",
                        "origin git@github.com:CompactAIOfficial/MythosMini.git",
                        "package source ssh://user3@example.com/user/repo",
                        "contact 'lane@example.com' for help",
                    ]
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "anonymized"
    result = runner.invoke(app, ["anonymize", str(input_dir), "--output", str(output_dir)])

    assert result.exit_code == 0
    text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "org.gnome.Shell@x11.service" in text
    assert "`@AGENT.md`" in text
    assert "@README.md" in text
    assert "-@plan.md" in text
    assert "https://medium.com/@ahmed.soliman/why-anthropics" in text
    assert "git@github.com:CompactAIOfficial/MythosMini.git" in text
    assert "ssh://user3@example.com/user/repo" in text
    assert "lane@example.com" not in text
    assert "user1@example.com" in text
    assert "email=1" in result.output


def test_pool_upload_is_reserved_until_backend_exists(tmp_path: Path):
    result = runner.invoke(app, ["pool", "upload", str(tmp_path)])

    assert result.exit_code == 1
    assert "not wired to a deployed pool backend yet" in result.output

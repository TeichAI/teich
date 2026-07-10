"""Deterministic anonymization for exported traces."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import os
import json
import re
import shutil
import string
import sys
from tempfile import NamedTemporaryFile
from typing import Any


TEXT_EXTENSIONS = {
    ".jsonl",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".log",
}

# ponytail: process startup isn't free — only fan out when there are enough
# files for the parallelism to pay for itself.
_MIN_FILES_FOR_PARALLEL = 8
_WINDOWS_MAX_PROCESS_WORKERS = 61


def _process_worker_count(file_count: int) -> int:
    workers = min(os.cpu_count() or 1, file_count)
    if sys.platform == "win32":
        workers = min(workers, _WINDOWS_MAX_PROCESS_WORKERS)
    return workers


@dataclass
class AnonymizeFileReport:
    path: Path
    output_path: Path
    replacements: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return any(count > 0 for count in self.replacements.values())


@dataclass
class AnonymizeReport:
    input_path: Path
    output_path: Path
    files: list[AnonymizeFileReport] = field(default_factory=list)

    @property
    def files_changed(self) -> int:
        return sum(1 for item in self.files if item.changed)

    @property
    def totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for item in self.files:
            for key, count in item.replacements.items():
                totals[key] = totals.get(key, 0) + count
        return totals


def anonymize_path(input_path: Path, output_path: Path, *, in_place: bool = False) -> AnonymizeReport:
    """Anonymize trace files under input_path."""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if in_place:
        output_path = input_path
    elif input_path.is_dir() and _path_contains(input_path, output_path):
        raise ValueError("Output path must not be inside the input directory")

    report = AnonymizeReport(input_path=input_path, output_path=output_path)
    if input_path.is_file():
        destination = output_path / input_path.name if output_path.exists() and output_path.is_dir() else output_path
        file_report = anonymize_file(input_path, destination)
        report.files.append(file_report)
        return report

    source_files = sorted(path for path in input_path.rglob("*") if path.is_file())
    destinations = [source if in_place else output_path / source.relative_to(input_path) for source in source_files]
    workers = _process_worker_count(len(source_files))
    if workers > 1 and len(source_files) >= _MIN_FILES_FOR_PARALLEL:
        # Each file is anonymized independently (fresh TraceAnonymizer per
        # file), so files can be processed in parallel safely.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            report.files.extend(executor.map(anonymize_file, source_files, destinations))
    else:
        for source_file, destination in zip(source_files, destinations):
            report.files.append(anonymize_file(source_file, destination))
    return report


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def anonymize_file(source: Path, destination: Path) -> AnonymizeFileReport:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() not in TEXT_EXTENSIONS:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return AnonymizeFileReport(path=source, output_path=destination)

    anonymizer = TraceAnonymizer()
    if source.suffix.lower() == ".jsonl":
        _anonymize_jsonl_file(source, destination, anonymizer)
    else:
        try:
            original = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return AnonymizeFileReport(path=source, output_path=destination)
        text = anonymizer.anonymize_text(original)
        if text != original or source.resolve() != destination.resolve():
            destination.write_text(text, encoding="utf-8")

    return AnonymizeFileReport(
        path=source,
        output_path=destination,
        replacements={key: value for key, value in anonymizer.counts.items() if value},
    )


def _anonymize_jsonl_file(source: Path, destination: Path, anonymizer: "TraceAnonymizer") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with (
            source.open("r", encoding="utf-8") as handle,
            NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_handle,
        ):
            temp_path = Path(temp_handle.name)
            changed = False
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    temp_handle.write(raw_line)
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError:
                    anonymized_line = anonymizer.anonymize_text(raw_line)
                    if anonymized_line != raw_line:
                        changed = True
                    temp_handle.write(anonymized_line)
                    continue
                anonymized = anonymizer.anonymize_value(value)
                if anonymized == value:
                    temp_handle.write(raw_line)
                    continue
                changed = True
                line = json.dumps(anonymized, ensure_ascii=False, separators=(",", ":"))
                line = line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
                temp_handle.write(line + "\n")
    except (OSError, UnicodeDecodeError):
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
        destination.write_text(
            anonymizer.anonymize_text(source.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
        return
    if temp_path is not None:
        if not changed and source.resolve() == destination.resolve():
            temp_path.unlink()
        else:
            temp_path.replace(destination)


class TraceAnonymizer:
    """Stateful per-trace anonymizer with consistent replacement maps."""

    _email_pattern = re.compile(
        r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9][A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9._%+-])"
    )
    _home_path_pattern = re.compile(
        r"(?P<prefix>(?:[A-Za-z]:)?[\\/]+(?:home|Users)[\\/]+)(?P<username>[^\\/:\s\"'<>|]+)",
        re.IGNORECASE,
    )
    _encoded_home_path_pattern = re.compile(
        r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>-(?:home|Users)-)(?P<username>[A-Za-z0-9._]+)(?=$|[-\\/\s\"'])",
        re.IGNORECASE,
    )
    _api_key_patterns = [
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>sk-or-v1-)(?P<body>[A-Za-z0-9_-]{16,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>sk-ant-api03-)(?P<body>[A-Za-z0-9_-]{16,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>sk-proj-)(?P<body>[A-Za-z0-9_-]{16,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>sk-)(?!or-v1-|ant-api03-|proj-)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>hf_)(?P<body>[A-Za-z0-9]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>gsk_)(?P<body>[A-Za-z0-9]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>github_pat_)(?P<body>[A-Za-z0-9_]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>gh[pousr]_)(?P<body>[A-Za-z0-9_]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>glpat-)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>lin_api_)(?P<body>[A-Za-z0-9]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>npm_)(?P<body>[A-Za-z0-9]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>pypi-)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>sk_(?:live|test)_)(?P<body>[A-Za-z0-9]{16,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>(?:rk|pk)_(?:live|test)_)(?P<body>[A-Za-z0-9]{16,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>whsec_)(?P<body>[A-Za-z0-9]{16,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>re_)(?P<body>[A-Za-z0-9]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>sq0(?:atp|csp)-)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>xox[baprs]-)(?P<body>[A-Za-z0-9-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>AIza)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>GOCSPX-)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>ctx7sk-)(?P<body>[A-Za-z0-9-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>(?:AKIA|ASIA))(?P<body>[A-Z0-9]{16})(?![A-Za-z0-9])"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>SK)(?P<body>[0-9a-fA-F]{32})(?![A-Za-z0-9])"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>SG\.)(?P<body>[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{20,})"),
    ]
    _jwt_pattern = re.compile(
        r"(?:(?<![A-Za-z0-9_-])|(?<=\\n))([A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})(?![A-Za-z0-9_-])"
    )
    _jwe_pattern = re.compile(
        r"(?:(?<![A-Za-z0-9_-])|(?<=\\n))([A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})(?![A-Za-z0-9_-])"
    )
    _bearer_pattern = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=-]{24,})")
    _generic_secret_pattern = re.compile(
        r"(?i)(\b(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|token|secret|password)\b\\?[\"']?[^\S\r\n]*[:=][^\S\r\n]*\\?[\"']?)([A-Za-z0-9_~+/=-]{24,})"
    )
    _quoted_assignment_pattern = re.compile(
        r"(?P<prefix>(?<![A-Za-z0-9_.-])(?P<key_quote>[\"']?)(?P<name>[A-Za-z][A-Za-z0-9_.-]{1,120})(?P=key_quote)[ \t]*(?:=|:)[ \t]*)(?P<quote>[\"'])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)"
    )
    _bare_assignment_pattern = re.compile(
        r"(?P<prefix>(?<![A-Za-z0-9_.-])(?P<key_quote>[\"']?)(?P<name>[A-Za-z][A-Za-z0-9_.-]{1,120})(?P=key_quote)[ \t]*(?:=|:)[ \t]*)(?P<value>[^\"'\s#]+)"
    )
    _credential_url_pattern = re.compile(
        r"(?i)\b(?P<scheme>[A-Za-z][A-Za-z0-9+.-]{1,32}://)(?P<user>[^:@/\s?#]*):(?P<password>[^@\s/?#]+)@"
    )
    _query_secret_pattern = re.compile(
        r"(?i)(?P<prefix>[?&](?P<name>[A-Za-z0-9_.-]*(?:api[_-]?key|token|secret|password|signature|sig|access[_-]?key|client[_-]?secret)[A-Za-z0-9_.-]*)=)(?P<value>[^&#\s\"'<>]+)"
    )
    _connection_component_secret_pattern = re.compile(
        r"(?i)(?P<prefix>(?:^|[;,\s])(?P<name>AccountKey|SharedAccessKey|SharedAccessSignature|Signature|Password|Pwd|AccessKeyId|SecretAccessKey|sig)=)(?P<value>[^;,\s]+)"
    )
    _private_key_block_pattern = re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    )
    _sensitive_exact_names = {
        "access_token",
        "admin_password",
        "anon_key",
        "api_key",
        "apikey",
        "app_secret",
        "auth_secret",
        "auth_token",
        "aws_secret_access_key",
        "azure_storage_connection_string",
        "broker_url",
        "celery_broker_url",
        "client_secret",
        "connection_string",
        "cookie_secret",
        "dashboard_password",
        "database_dsn",
        "database_url",
        "db_password",
        "db_url",
        "direct_url",
        "django_secret_key",
        "dsn",
        "encryption_key",
        "enc_key",
        "firebase_private_key",
        "jwt_secret",
        "mail_password",
        "mail_url",
        "mongo_uri",
        "mongo_url",
        "mongodb_uri",
        "mongodb_url",
        "mysql_password",
        "mysql_url",
        "nextauth_secret",
        "password",
        "pgpassword",
        "postgres_password",
        "postgres_url",
        "postgresql_url",
        "private_key",
        "rabbitmq_url",
        "redis_password",
        "redis_url",
        "refresh_token",
        "root_password",
        "secret",
        "secret_key",
        "secret_key_base",
        "sentry_dsn",
        "service_role_key",
        "session_secret",
        "signing_secret",
        "smtp_password",
        "smtp_url",
        "ssh_private_key",
        "supabase_anon_key",
        "supabase_service_role_key",
        "token",
        "vault_enc_key",
        "webhook_secret",
    }
    _sensitive_name_words = {
        "auth",
        "bearer",
        "credential",
        "credentials",
        "jwt",
        "pass",
        "passwd",
        "password",
        "private",
        "pwd",
        "secret",
        "signature",
        "token",
    }
    _sensitive_key_context_words = {
        "access",
        "account",
        "admin",
        "anon",
        "api",
        "app",
        "auth",
        "aws",
        "azure",
        "client",
        "dashboard",
        "enc",
        "encryption",
        "firebase",
        "google",
        "master",
        "private",
        "publishable",
        "root",
        "role",
        "secret",
        "service",
        "signing",
        "supabase",
        "vault",
    }
    _sensitive_url_context_words = {
        "amqp",
        "broker",
        "celery",
        "database",
        "db",
        "direct",
        "dsn",
        "mail",
        "mariadb",
        "mongo",
        "mongodb",
        "mysql",
        "postgres",
        "postgresql",
        "rabbitmq",
        "redis",
        "sentry",
        "smtp",
    }
    _non_secret_name_words = {
        "count",
        "enabled",
        "enable",
        "expiry",
        "expires",
        "has",
        "length",
        "limit",
        "max",
        "min",
        "num",
        "number",
        "supports",
        "tokenizer",
        "tokens",
        "ttl",
        "use",
        "uses",
    }
    _empty_secret_values = {
        "",
        "false",
        "nil",
        "none",
        "null",
        "true",
        "undefined",
    }
    _non_person_usernames = {
        "all users",
        "default",
        "defaultuser0",
        "public",
        "shared",
    }
    _systemd_unit_suffixes = (
        ".automount",
        ".device",
        ".mount",
        ".path",
        ".scope",
        ".service",
        ".slice",
        ".socket",
        ".swap",
        ".target",
        ".timer",
    )
    _known_git_remote_addresses = {
        "git@github.com",
        "git@ssh.github.com",
        "git@gitlab.com",
        "git@bitbucket.org",
    }
    _non_email_tlds = {
        "automount",
        "device",
        "local",
        "localhost",
        "module",
        "mount",
        "path",
        "scope",
        "service",
        "slice",
        "socket",
        "swap",
        "target",
        "test",
        "timer",
    }

    # ponytail: literal gates — each entry pairs the regexes above with cheap
    # substring checks so the expensive lookbehind patterns only run on text
    # that could possibly match. Order mirrors _api_key_patterns.
    _api_key_gates = (
        ("sk-or-v1-",),
        ("sk-ant-api03-",),
        ("sk-proj-",),
        ("sk-",),
        ("hf_",),
        ("gsk_",),
        ("github_pat_",),
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        ("glpat-",),
        ("lin_api_",),
        ("npm_",),
        ("pypi-",),
        ("sk_live_", "sk_test_"),
        ("rk_live_", "rk_test_", "pk_live_", "pk_test_"),
        ("whsec_",),
        ("re_",),
        ("sq0atp-", "sq0csp-"),
        ("xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"),
        ("AIza",),
        ("GOCSPX-",),
        ("ctx7sk-",),
        ("AKIA", "ASIA"),
        ("SK",),
        ("SG.",),
    )
    # Every name _is_sensitive_name can fire on contains one of these
    # substrings (lowercased), so assignment/generic-secret scans are skipped
    # when none is present.
    _sensitive_gate_words = (
        "pass",
        "pwd",
        "secret",
        "credential",
        "private",
        "token",
        "sig",
        "jwt",
        "key",
        "url",
        "uri",
        "dsn",
        "connection",
    )
    _jwt_gate = re.compile(r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")

    def __init__(self) -> None:
        self.counts = {"email": 0, "username": 0, "api_key": 0}
        self._email_map: dict[str, str] = {}
        self._username_map: dict[str, str] = {}
        self._api_key_map: dict[str, str] = {}
        self._api_replacements: set[str] = set()
        self._owner_group_patterns: dict[str, re.Pattern[str] | None] = {}

    def anonymize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.anonymize_text(value)
        if isinstance(value, list):
            return [self.anonymize_value(item) for item in value]
        if isinstance(value, dict):
            return self._anonymize_mapping(value)
        return value

    def _anonymize_mapping(self, value: dict[Any, Any]) -> dict[Any, Any]:
        should_preserve_base64_data = self._looks_like_base64_media_source(value)
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            redacted_key = self.anonymize_value(key)
            if (
                should_preserve_base64_data
                and key == "data"
                and isinstance(item, str)
                and self._looks_like_base64_blob(item)
            ):
                redacted[redacted_key] = item
            elif isinstance(key, str) and self._should_redact_mapping_value(key, item):
                redacted[redacted_key] = self._redact_mapping_value(key, item)
            else:
                redacted[redacted_key] = self.anonymize_value(item)
        return redacted

    def _should_redact_mapping_value(self, key: str, item: Any) -> bool:
        if isinstance(item, str):
            return self._should_redact_assignment(key, item)
        if item is None or isinstance(item, bool):
            return False
        if isinstance(item, int | float):
            return self._is_sensitive_name(key, str(item))
        return False

    def _redact_mapping_value(self, key: str, item: Any) -> str:
        value = str(item)
        return self._assignment_secret_replacement(key, value)

    @staticmethod
    def _looks_like_base64_media_source(value: dict[Any, Any]) -> bool:
        source_type = value.get("type")
        media_type = value.get("media_type") or value.get("mime_type")
        return (
            isinstance(source_type, str)
            and source_type.lower() == "base64"
            and isinstance(value.get("data"), str)
        ) or (
            isinstance(media_type, str)
            and media_type.lower().startswith(("image/", "audio/", "video/"))
            and isinstance(value.get("data"), str)
        )

    @staticmethod
    def _looks_like_base64_blob(value: str) -> bool:
        if len(value) < 256:
            return False
        return re.fullmatch(r"[A-Za-z0-9+/=\s]+", value) is not None

    def anonymize_text(self, text: str) -> str:
        lowered = text.lower()
        if "@" in text:
            text = self._replace_emails(text)
        if "home" in lowered or "users" in lowered:
            text = self._replace_usernames_in_paths(text)
            text = self._replace_encoded_home_usernames(text)
        if self._username_map:
            text = self._replace_known_unix_owner_group_usernames(text)
        text = self._replace_api_keys(text)
        return text

    def _replace_emails(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            original = match.group(1)
            if self._looks_like_non_email_reference(original, match, text):
                return original
            key = original.lower()
            if key not in self._email_map:
                local_part = original.split("@", maxsplit=1)[0]
                username = self._map_username(local_part)
                self._email_map[key] = f"redacted-{username}@example.com"
            self.counts["email"] += 1
            return self._email_map[key]

        return self._email_pattern.sub(replace, text)

    def _looks_like_non_email_reference(self, value: str, match: re.Match[str], text: str) -> bool:
        if match.start() > 0 and text[match.start() - 1] == "\\":
            return True
        if "`" in value:
            return True
        if "/" in value or "\\" in value:
            return True
        if self._looks_like_uri_userinfo(match, text):
            return True
        lowered = value.lower()
        if lowered.endswith(self._systemd_unit_suffixes):
            return True
        domain = lowered.rsplit("@", maxsplit=1)[-1]
        tld = domain.rsplit(".", maxsplit=1)[-1]
        if tld in self._non_email_tlds:
            return True
        if lowered in self._known_git_remote_addresses:
            return True
        if self._looks_like_path_or_package_reference(match, text):
            return True
        return False

    @staticmethod
    def _looks_like_uri_userinfo(match: re.Match[str], text: str) -> bool:
        start = match.start()
        segment_start = max(text.rfind(boundary, 0, start) for boundary in " \t\r\n\"'<>")
        prefix_segment = text[segment_start + 1:start]
        scheme_separator = prefix_segment.rfind("://")
        if scheme_separator == -1:
            return False
        scheme = prefix_segment[:scheme_separator].rsplit("/", maxsplit=1)[-1]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*", scheme):
            return False
        authority_prefix = prefix_segment[scheme_separator + 3:]
        return not any(separator in authority_prefix for separator in "/\\?#")

    @staticmethod
    def _looks_like_path_or_package_reference(match: re.Match[str], text: str) -> bool:
        start = match.start()
        end = match.end()
        previous_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""
        if previous_char and previous_char in "/\\":
            return True
        if next_char and next_char in "/\\":
            return True

        segment_start = max(text.rfind(boundary, 0, start) for boundary in " \t\r\n\"'<>")
        segment_end_candidates = [index for index in (text.find(boundary, end) for boundary in " \t\r\n\"'<>") if index >= 0]
        segment_end = min(segment_end_candidates) if segment_end_candidates else len(text)
        segment = text[segment_start + 1:segment_end]
        if re.search(r"[/\\]", segment):
            return True
        if re.search(r"[@:]v?\d+(?:\.\d+){1,}", segment, re.IGNORECASE):
            return True
        return False

    def _replace_usernames_in_paths(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            username = match.group("username")
            if username.lower() in self._non_person_usernames:
                return match.group(0)
            self.counts["username"] += 1
            return f"{match.group('prefix')}{self._map_username(username)}"

        return self._home_path_pattern.sub(replace, text)

    def _replace_encoded_home_usernames(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            username = match.group("username")
            if username.lower() in self._non_person_usernames:
                return match.group(0)
            self.counts["username"] += 1
            return f"{match.group('prefix')}{self._map_username(username)}"

        return self._encoded_home_path_pattern.sub(replace, text)

    def _replace_known_unix_owner_group_usernames(self, text: str) -> str:
        for username, replacement in list(self._username_map.items()):
            if username not in self._owner_group_patterns:
                if not re.fullmatch(r"[A-Za-z0-9._-]{3,}", username) or username in self._non_person_usernames:
                    self._owner_group_patterns[username] = None
                else:
                    self._owner_group_patterns[username] = re.compile(
                        rf"(?<!\S){re.escape(username)}\s+{re.escape(username)}(?=\s+\d)",
                        re.IGNORECASE,
                    )
            pattern = self._owner_group_patterns[username]
            if pattern is None or username.lower() not in text.lower():
                continue
            text, count = pattern.subn(f"{replacement} {replacement}", text)
            if count:
                self.counts["username"] += count * 2
        return text

    def _replace_api_keys(self, text: str) -> str:
        if "-----BEGIN " in text:
            text = self._private_key_block_pattern.sub(self._replace_private_key_block, text)
        for gates, pattern in zip(self._api_key_gates, self._api_key_patterns):
            if any(gate in text for gate in gates):
                text = pattern.sub(self._replace_prefixed_key, text)
        if self._jwt_gate.search(text):
            text = self._jwe_pattern.sub(self._replace_jwt, text)
            text = self._jwt_pattern.sub(self._replace_jwt, text)
        lowered = text.lower()
        if "bearer" in lowered:
            text = self._bearer_pattern.sub(self._replace_bearer, text)
        if "://" in text and "@" in text:
            text = self._credential_url_pattern.sub(self._replace_credential_url_password, text)
        if any(word in lowered for word in self._sensitive_gate_words):
            if "=" in text and ("?" in text or "&" in text):
                text = self._query_secret_pattern.sub(self._replace_query_secret, text)
            if "=" in text:
                text = self._connection_component_secret_pattern.sub(self._replace_connection_component_secret, text)
            if "=" in text or ":" in text:
                text = self._quoted_assignment_pattern.sub(self._replace_sensitive_quoted_assignment, text)
                text = self._bare_assignment_pattern.sub(self._replace_sensitive_bare_assignment, text)
                text = self._generic_secret_pattern.sub(self._replace_generic_secret, text)
        return text

    def _map_username(self, username: str) -> str:
        key = username.strip().lower()
        if key not in self._username_map:
            self._username_map[key] = f"user{len(self._username_map) + 1}"
        return self._username_map[key]

    def _replace_prefixed_key(self, match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = "redacted_api_key_" + self._dummy_sequence(token, 16)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        self.counts["api_key"] += 1
        return replacement

    def _replace_private_key_block(self, match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = "redacted_private_key_" + self._dummy_sequence(token, 16)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        self.counts["api_key"] += 1
        return replacement

    def _replace_jwt(self, match: re.Match[str]) -> str:
        token = match.group(1)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = "redacted_jwt_" + self._dummy_sequence(token, 16)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        self.counts["api_key"] += 1
        return replacement

    def _replace_bearer(self, match: re.Match[str]) -> str:
        token = match.group(2)
        if token in self._api_replacements:
            return match.group(0)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = "redacted_" + self._dummy_sequence(token, 24)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        self.counts["api_key"] += 1
        return match.group(1) + replacement

    def _replace_credential_url_password(self, match: re.Match[str]) -> str:
        password = match.group("password")
        if self._is_redacted_secret_value(password):
            return match.group(0)
        replacement = self._secret_replacement(password)
        self.counts["api_key"] += 1
        return f"{match.group('scheme')}{match.group('user')}:{replacement}@"

    def _replace_query_secret(self, match: re.Match[str]) -> str:
        value = match.group("value")
        if self._is_redacted_secret_value(value):
            return match.group(0)
        replacement = self._secret_replacement(f"{match.group('name')}={value}")
        self.counts["api_key"] += 1
        return match.group("prefix") + replacement

    def _replace_connection_component_secret(self, match: re.Match[str]) -> str:
        value = match.group("value")
        if self._is_redacted_secret_value(value):
            return match.group(0)
        replacement = self._secret_replacement(f"{match.group('name')}={value}")
        self.counts["api_key"] += 1
        return match.group("prefix") + replacement

    def _replace_sensitive_quoted_assignment(self, match: re.Match[str]) -> str:
        name = match.group("name")
        value = match.group("value")
        if not self._should_redact_assignment(name, value):
            return match.group(0)
        replacement = self._assignment_secret_replacement(name, value)
        return f"{match.group('prefix')}{match.group('quote')}{replacement}{match.group('quote')}"

    def _replace_sensitive_bare_assignment(self, match: re.Match[str]) -> str:
        name = match.group("name")
        value, suffix = self._split_bare_assignment_value(match.group("value"))
        if not self._should_redact_assignment(name, value):
            return match.group(0)
        replacement = self._assignment_secret_replacement(name, value)
        return match.group("prefix") + replacement + suffix

    def _replace_generic_secret(self, match: re.Match[str]) -> str:
        token = match.group(2)
        if token in self._api_replacements:
            return match.group(0)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = "redacted_" + self._dummy_sequence(token, 24)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        self.counts["api_key"] += 1
        return match.group(1) + replacement

    def _assignment_secret_replacement(self, name: str, value: str) -> str:
        if self._private_key_block_pattern.search(value):
            replacement = self._api_key_map.get(value)
            if replacement is None:
                replacement = "redacted_private_key_" + self._dummy_sequence(value, 16)
                self._api_key_map[value] = replacement
                self._api_replacements.add(replacement)
            self.counts["api_key"] += 1
            return replacement
        if self._looks_like_jwt(value):
            replacement = self._api_key_map.get(value)
            if replacement is None:
                replacement = "redacted_jwt_" + self._dummy_sequence(value, 16)
                self._api_key_map[value] = replacement
                self._api_replacements.add(replacement)
            self.counts["api_key"] += 1
            return replacement
        replacement = self._secret_replacement(f"{name}={value}")
        self.counts["api_key"] += 1
        return replacement

    def _secret_replacement(self, token: str) -> str:
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = "redacted_secret_" + self._dummy_sequence(token, 16)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        return replacement

    def _should_redact_assignment(self, name: str, value: str) -> bool:
        if not value or self._is_empty_secret_value(value) or self._is_redacted_secret_value(value):
            return False
        return self._is_sensitive_name(name, value)

    def _is_sensitive_name(self, name: str, value: str) -> bool:
        normalized = self._normalize_secret_name(name)
        words = self._secret_name_words(name)
        if normalized in self._sensitive_exact_names:
            return True
        if normalized.endswith("_connection_string") or "connection_string" in normalized:
            return self._looks_like_secret_value(value) or len(value.strip()) >= 8
        if words & {"password", "passwd", "pwd", "secret", "credential", "credentials", "private"}:
            return True
        if "token" in words and not words & self._non_secret_name_words:
            return True
        if "signature" in words or "sig" in words:
            return True
        if "jwt" in words:
            return bool(words & {"secret", "token", "key", "auth"}) or self._looks_like_jwt(value)
        if "key" in words:
            return bool(words & self._sensitive_key_context_words) or self._looks_like_secret_value(value)
        if words & {"url", "uri", "dsn"}:
            return bool(words & self._sensitive_url_context_words) or self._looks_like_credential_url(value)
        return False

    @classmethod
    def _normalize_secret_name(cls, name: str) -> str:
        return "_".join(cls._secret_name_words(name))

    @staticmethod
    def _secret_name_words(name: str) -> set[str]:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
        return {part for part in normalized.split("_") if part}

    @staticmethod
    def _split_bare_assignment_value(value: str) -> tuple[str, str]:
        suffix = ""
        while value and value[-1] in ",;":
            suffix = value[-1] + suffix
            value = value[:-1]
        return value, suffix

    def _looks_like_secret_value(self, value: str) -> bool:
        stripped = value.strip()
        if self._is_empty_secret_value(stripped) or self._is_redacted_secret_value(stripped):
            return False
        if self._looks_like_jwt(stripped) or self._looks_like_credential_url(stripped):
            return True
        if self._private_key_block_pattern.search(stripped):
            return True
        for pattern in self._api_key_patterns:
            if pattern.search(stripped):
                return True
        if len(stripped) >= 20 and re.fullmatch(r"[A-Za-z0-9_~+/=.:-]+", stripped):
            return True
        return False

    def _looks_like_jwt(self, value: str) -> bool:
        return self._jwt_pattern.fullmatch(value.strip()) is not None or self._jwe_pattern.fullmatch(value.strip()) is not None

    def _looks_like_credential_url(self, value: str) -> bool:
        return self._credential_url_pattern.search(value) is not None

    @classmethod
    def _is_empty_secret_value(cls, value: str) -> bool:
        stripped = value.strip().strip("\"'")
        return stripped.lower() in cls._empty_secret_values

    @staticmethod
    def _is_redacted_secret_value(value: str) -> bool:
        stripped = value.strip().strip("\"'")
        return stripped.startswith(("redacted_", "<redacted", "[redacted"))

    @staticmethod
    def _dummy_sequence(seed: str, length: int) -> str:
        alphabet = string.ascii_letters + string.digits
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        chars: list[str] = []
        counter = 0
        while len(chars) < length:
            block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
            chars.extend(alphabet[byte % len(alphabet)] for byte in block)
            counter += 1
        return "".join(chars[:length])

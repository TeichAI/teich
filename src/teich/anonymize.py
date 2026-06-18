"""Deterministic anonymization for exported traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re
import shutil
import string
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
        file_report = _anonymize_file(input_path, destination)
        report.files.append(file_report)
        return report

    for source_file in sorted(path for path in input_path.rglob("*") if path.is_file()):
        relative_path = source_file.relative_to(input_path)
        destination = source_file if in_place else output_path / relative_path
        file_report = _anonymize_file(source_file, destination)
        report.files.append(file_report)
    return report


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _anonymize_file(source: Path, destination: Path) -> AnonymizeFileReport:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() not in TEXT_EXTENSIONS:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return AnonymizeFileReport(path=source, output_path=destination)

    anonymizer = TraceAnonymizer()
    if source.suffix.lower() == ".jsonl":
        text = _anonymize_jsonl_text(source, anonymizer)
    else:
        try:
            original = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return AnonymizeFileReport(path=source, output_path=destination)
        text = anonymizer.anonymize_text(original)

    destination.write_text(text, encoding="utf-8")
    return AnonymizeFileReport(
        path=source,
        output_path=destination,
        replacements={key: value for key, value in anonymizer.counts.items() if value},
    )


def _anonymize_jsonl_text(source: Path, anonymizer: "TraceAnonymizer") -> str:
    rows: list[str] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    rows.append(raw_line)
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError:
                    rows.append(anonymizer.anonymize_text(raw_line))
                    continue
                anonymized = anonymizer.anonymize_value(value)
                if anonymized == value:
                    rows.append(raw_line)
                    continue
                line = json.dumps(anonymized, ensure_ascii=False, separators=(",", ":"))
                line = line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
                rows.append(line + "\n")
    except (OSError, UnicodeDecodeError):
        return anonymizer.anonymize_text(source.read_text(encoding="utf-8", errors="replace"))
    return "".join(rows)


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
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>xox[baprs]-)(?P<body>[A-Za-z0-9-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>AIza)(?P<body>[A-Za-z0-9_-]{20,})"),
        re.compile(r"(?:(?<![A-Za-z0-9])|(?<=\\n))(?P<prefix>ctx7sk-)(?P<body>[A-Za-z0-9-]{20,})"),
    ]
    _jwt_pattern = re.compile(
        r"(?:(?<![A-Za-z0-9_-])|(?<=\\n))(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})(?![A-Za-z0-9_-])"
    )
    _bearer_pattern = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=-]{24,})")
    _generic_secret_pattern = re.compile(
        r"(?i)(\b(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|token|secret|password)\b\\?[\"']?[^\S\r\n]*[:=][^\S\r\n]*\\?[\"']?)([A-Za-z0-9_~+/=-]{24,})"
    )
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

    def __init__(self) -> None:
        self.counts = {"email": 0, "username": 0, "api_key": 0}
        self._email_map: dict[str, str] = {}
        self._username_map: dict[str, str] = {}
        self._api_key_map: dict[str, str] = {}
        self._api_replacements: set[str] = set()

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
            else:
                redacted[redacted_key] = self.anonymize_value(item)
        return redacted

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
        text = self._replace_emails(text)
        text = self._replace_usernames_in_paths(text)
        text = self._replace_encoded_home_usernames(text)
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
                self._email_map[key] = f"{username}@example.com"
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
        if lowered in self._known_git_remote_addresses:
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
            if not re.fullmatch(r"[A-Za-z0-9._-]{3,}", username):
                continue
            if username in self._non_person_usernames:
                continue
            pattern = re.compile(
                rf"(?<!\S){re.escape(username)}\s+{re.escape(username)}(?=\s+\d)",
                re.IGNORECASE,
            )
            text, count = pattern.subn(f"{replacement} {replacement}", text)
            if count:
                self.counts["username"] += count * 2
        return text

    def _replace_api_keys(self, text: str) -> str:
        for pattern in self._api_key_patterns:
            text = pattern.sub(self._replace_prefixed_key, text)
        text = self._jwt_pattern.sub(self._replace_jwt, text)
        text = self._bearer_pattern.sub(self._replace_bearer, text)
        text = self._generic_secret_pattern.sub(self._replace_generic_secret, text)
        return text

    def _map_username(self, username: str) -> str:
        key = username.strip().lower()
        if key not in self._username_map:
            self._username_map[key] = f"user{len(self._username_map) + 1}"
        return self._username_map[key]

    def _replace_prefixed_key(self, match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        token = match.group(0)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = prefix + self._dummy_sequence(token, 32)
            self._api_key_map[token] = replacement
            self._api_replacements.add(replacement)
        self.counts["api_key"] += 1
        return replacement

    def _replace_jwt(self, match: re.Match[str]) -> str:
        token = match.group(1)
        replacement = self._api_key_map.get(token)
        if replacement is None:
            replacement = ".".join(
                [
                    "eyJ0eXAiOiJKV1Qi",
                    self._dummy_sequence(token + ".payload", 24),
                    self._dummy_sequence(token + ".signature", 32),
                ]
            )
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

"""Tests for bench mode: config (sources array), the shared backend base/harvest, the
harbor backend, and the driver loop. Live harbor/Docker runs stay integration-only; here we
cover config, reward/route/harvest, harbor resolution/normalization, and driver behavior."""

import json

import pytest

from teich.bench import run_bench
from teich.bench import runner as bench_runner
from teich.bench.backends import base, get_backend
from teich.bench.backends import harbor as hb
from teich.config import BenchSource, Config


# --------------------------------------------------------------------------- config

def test_bench_config_defaults():
    assert Config().bench.sources == []


def test_bench_sources_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "bench:\n"
        "  sources:\n"
        "    - { type: harbor, source: terminal-bench@2.0 }\n"
        "    - { type: swe-bench, source: SWE-bench/SWE-bench_Verified, split: test }\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(config_file)
    assert [s.type for s in cfg.bench.sources] == ["harbor", "swe-bench"]
    assert cfg.bench.sources[0].source == "terminal-bench@2.0"
    assert cfg.bench.sources[0].backend == "docker"          # default
    assert cfg.bench.sources[1].split == "test"


def test_run_bench_requires_sources():
    with pytest.raises(RuntimeError, match="bench.sources"):
        run_bench(Config())


def test_get_backend():
    assert get_backend("harbor").type == "harbor"
    with pytest.raises(RuntimeError, match="Unknown bench source type"):
        get_backend("nope")


# ------------------------------------------------------------------- base: scoring/routing

def test_numeric_primary_route():
    assert base.numeric(1) == 1.0 and base.numeric(True) is None and base.numeric("x") is None
    assert base.primary_score({"reward": 1.0, "sub": 0.5}) == 1.0   # 'reward' preferred
    assert base.primary_score({"score": 0.6}) == 0.6                # else first numeric
    assert base.primary_score(None) is None
    assert base.route_split(1.0) == "passed"
    assert base.route_split(0.0) == "failed"
    assert base.route_split(None) == "failed"
    assert base.route_split(0.6) == "borderline"


def test_rewards_from_mapping_keeps_full_dict():
    assert base.rewards_from_mapping({"rewards": {"reward": 1.0, "sub": 0.5, "bad": "x"}}) == {
        "reward": 1.0,
        "sub": 0.5,
    }
    assert base.rewards_from_mapping({"rewards": {}}) is None
    assert base.rewards_from_mapping(None) is None


def test_bench_stem_namespaced_by_source():
    s = BenchSource(type="harbor", source="terminal-bench@2.0")
    assert base.bench_stem(s, "task-a") == "bench-terminal-bench-2.0-task-a"
    s2 = BenchSource(type="swe-bench", source="SWE-bench/SWE-bench_Verified")
    assert base.bench_stem(s2, "astropy__astropy-12907") == (
        "bench-SWE-bench-SWE-bench_Verified-astropy__astropy-12907"
    )


def test_source_id_distinguishes_by_knobs():
    # Same `source` but a differing knob (split) must yield distinct ids/stems so the two
    # sources can't overwrite each other or wrongly resume-skip; the plain case is unchanged.
    a = BenchSource(type="swe-bench", source="ds", split="train")
    b = BenchSource(type="swe-bench", source="ds", split="test")
    assert base.source_id(a) != base.source_id(b)
    assert base.bench_stem(a, "t") != base.bench_stem(b, "t")
    assert base.source_id(BenchSource(type="swe-bench", source="ds")) == "ds"


def test_source_id_bounds_large_instance_list():
    # A big instances subset must not leak its full comma-joined list into source_id (and thus
    # into filenames / Docker tags / container names past OS/Docker length limits).
    many = [f"repo__proj-{i:04d}" for i in range(30)]
    src = BenchSource(type="swe-bench", source="ds", instances=many)
    sid = base.source_id(src)
    assert len(sid) < 60 and "repo__proj-0029" not in sid
    # order-independent (listing order shouldn't spawn a distinct id / dataset)
    assert base.source_id(BenchSource(type="swe-bench", source="ds", instances=list(reversed(many)))) == sid
    # distinct instance sets stay distinct
    assert base.source_id(BenchSource(type="swe-bench", source="ds", instances=many[:15])) != sid
    # a single instance stays human-readable
    assert "django__django-1" in base.source_id(
        BenchSource(type="swe-bench", source="ds", instances=["django__django-1"])
    )


def test_bench_source_namespace_field():
    # swe-bench image namespace: default "swebench" pulls published images; null builds custom
    # instances locally. (The pull/build execution itself is Docker/integration-only.)
    assert BenchSource(type="swe-bench", source="ds").namespace == "swebench"
    assert BenchSource(type="swe-bench", source="ds", namespace=None).namespace is None


def test_bench_progress_tracks_counts_when_disabled():
    # No terminal -> no live bar, but the pass/fail/borderline/error tally is still tracked.
    p = bench_runner._BenchProgress(console=None)
    assert not p.enabled
    with p:
        bar = p.add_source("swe: ds", 4)
        assert bar is None
        p.advance(bar, split="passed")
        p.advance(bar, split="borderline")
        p.advance(bar, split="failed")
        p.advance(bar, errored=True)
    assert (p.passed, p.failed, p.borderline, p.errored) == (1, 1, 1, 1)


def test_bench_progress_renders_with_terminal_console():
    # A forced-terminal console builds the rich Progress; exercises the column/field wiring
    # (e.g. the `tally` field must be supplied to add_task) without asserting rendered output.
    from rich.console import Console

    p = bench_runner._BenchProgress(Console(force_terminal=True))
    assert p.enabled
    with p:
        bar = p.add_source("swe: ds", 2)
        assert bar is not None
        p.advance(bar, split="passed")
        p.advance(bar, errored=True)
    assert (p.passed, p.errored) == (1, 1)


def test_harbor_source_slug_keys_on_repo():
    # Same spec/version but different repo must not share a download cache dir (else the second
    # source silently reuses the first repo's tasks). Absent repo, the key is unchanged.
    assert hb._source_slug("ds", "1.0") == hb._source_slug("ds", "1.0", None)
    assert hb._source_slug("ds", "1.0", "orgA/reg") != hb._source_slug("ds", "1.0", "orgB/reg")
    assert hb._source_slug("ds", "1.0", "orgA/reg") != hb._source_slug("ds", "1.0")


def test_run_bench_rejects_codex_host_auth(tmp_path, monkeypatch):
    # Bench backends don't wire Codex host auth into the task container; fail fast rather than
    # launch a silently-unauthenticated run when host auth is the only configured credential.
    for var in ("TEICH_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = Config(
        agent={"provider": "codex", "codex": {"use_host_auth": True}},
        bench={"sources": [{"type": "harbor", "source": "terminal-bench@2.0"}]},
        output={"traces_dir": tmp_path / "o"},
    )
    assert cfg.get_api_key() is None
    with pytest.raises(RuntimeError, match="does not yet support Codex host auth"):
        run_bench(cfg)


def test_run_bench_rejects_chat_wire_api_on_custom_endpoint(tmp_path):
    # Bench doesn't thread api.wire_api into pi/codex, so a chat-completions-only endpoint (e.g.
    # z.ai via provider: openai + base_url) would silently be hit on /responses. Fail fast.
    def _cfg(**api):
        return Config(
            agent={"provider": "pi"},
            api=api,
            bench={"sources": [{"type": "harbor", "source": "S"}]},
            output={"traces_dir": tmp_path / "o"},
        )

    with pytest.raises(RuntimeError, match="does not yet thread api.wire_api"):
        run_bench(_cfg(provider="openai", base_url="https://api.z.ai/api/paas/v4",
                       wire_api="chat_completions", api_key="k"))
    # openrouter is exempt (its prefix routing already uses completions); responses is fine.
    #  (these get past the guard and fail later for unrelated reasons — assert the guard message is absent)
    for ok in (dict(provider="openrouter", wire_api="chat_completions", api_key="k"),
               dict(provider="openai", wire_api="responses", api_key="k")):
        try:
            run_bench(_cfg(**ok))
        except RuntimeError as exc:
            assert "does not yet thread api.wire_api" not in str(exc)


def test_existing_output_across_splits(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    assert base.existing_output(cfg, "bench-x") is None
    routed = tmp_path / "output" / "borderline" / "bench-x.jsonl"
    routed.parent.mkdir(parents=True)
    routed.write_text("{}\n", encoding="utf-8")
    assert base.existing_output(cfg, "bench-x") == routed


def test_bench_root_sibling_of_output(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "out"})
    assert base.bench_root(cfg) == tmp_path / "bench"
    cfg2 = Config(output={"traces_dir": tmp_path / "out", "bench_dir": tmp_path / "custom"})
    assert base.bench_root(cfg2) == tmp_path / "custom"


def test_bench_root_rejects_in_tree_bench_dir(tmp_path):
    # A bench_dir inside traces_dir would get uploaded + misclassified as dataset rows.
    with pytest.raises(RuntimeError, match="must be outside"):
        base.bench_root(Config(output={"traces_dir": tmp_path / "out", "bench_dir": tmp_path / "out" / "bench"}))
    with pytest.raises(RuntimeError, match="must be outside"):
        base.bench_root(Config(output={"traces_dir": tmp_path / "out", "bench_dir": tmp_path / "out"}))


def test_bench_root_default_rejects_output_named_bench(tmp_path):
    # --output bench makes the computed default sibling (traces_dir.parent/"bench") collide with
    # the dataset dir; the default must be guarded too, not just an explicit output.bench_dir.
    with pytest.raises(RuntimeError, match="must be outside"):
        base.bench_root(Config(output={"traces_dir": tmp_path / "bench"}))


# ------------------------------------------------------------------- base: harvest

def test_harvest_writes_native_trace_and_metadata(tmp_path):
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source="terminal-bench@2.0")
    task = base.BenchTask(id="add-bug")
    run = base.BenchRun(
        native_lines=['{"type":"session","id":"s"}', '{"type":"message","message":{"role":"user","content":[]}}'],
        rewards={"reward": 0.6, "tests": 0.8},
        metadata={"model": "z-ai/glm-5.2", "exception": None},
    )
    paths, split = base.harvest(cfg, source, task, run)
    stem = base.bench_stem(source, "add-bug")
    assert split == "borderline"
    assert paths == [tmp_path / "output" / "borderline" / f"{stem}.jsonl"]
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row.get("type") for row in rows} == {"session", "message"}  # native, not converted
    meta = json.loads((tmp_path / "output" / "metadata" / f"{stem}.json").read_text(encoding="utf-8"))
    assert meta["split"] == "borderline" and meta["reward"] == 0.6
    assert meta["rewards"] == {"reward": 0.6, "tests": 0.8}            # full dict, no clamping
    assert meta["source"] == "terminal-bench-2.0" and meta["type"] == "harbor"
    assert meta["agent"] == "pi" and meta["model"] == "z-ai/glm-5.2"


def test_harvest_reroute_removes_stale_split_copy(tmp_path):
    # Re-harvesting a task (no --resume) whose score crosses a routing boundary must not leave a
    # stale copy in the old split — the dataset scanners read every split.
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source="ds")
    task = base.BenchTask(id="t")
    stem = base.bench_stem(source, "t")
    base.harvest(cfg, source, task, base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 0.0}))
    assert (tmp_path / "output" / "failed" / f"{stem}.jsonl").is_file()

    paths, split = base.harvest(
        cfg, source, task, base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0})
    )
    assert split == "passed"
    assert (tmp_path / "output" / "passed" / f"{stem}.jsonl").is_file()
    assert not (tmp_path / "output" / "failed" / f"{stem}.jsonl").exists()
    meta = json.loads((tmp_path / "output" / "metadata" / f"{stem}.json").read_text(encoding="utf-8"))
    assert meta["split"] == "passed"


# ------------------------------------------------------------------- harbor backend helpers

def test_harbor_agent_name_mapping():
    assert hb._agent_name_for("codex") == "codex"
    assert hb._agent_name_for("claude-code") == "claude-code"
    assert hb._agent_name_for("claude") == "claude-code"
    assert hb._agent_name_for("pi") == "pi"
    assert hb._agent_name_for("hermes") == "hermes"
    with pytest.raises(RuntimeError, match="does not support agent provider"):
        hb._agent_name_for("chat")


def test_harbor_auth_env():
    env = hb._agent_auth_env(
        Config(api={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk"})
    )
    assert env["OPENAI_API_KEY"] == "sk" and env["OPENROUTER_API_KEY"] == "sk"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    env2 = hb._agent_auth_env(Config(api={"provider": "openai", "api_key": "sk-o"}))
    assert env2["OPENAI_API_KEY"] == "sk-o" and "OPENROUTER_API_KEY" not in env2


def test_harbor_model_prefix():
    pi_or = Config(agent={"provider": "pi"}, model={"model": "z-ai/glm-5.2"}, api={"provider": "openrouter"})
    assert hb._bench_model_name(pi_or) == "openrouter/z-ai/glm-5.2"
    already = Config(agent={"provider": "pi"}, model={"model": "openrouter/z-ai/glm-5.2"}, api={"provider": "openrouter"})
    assert hb._bench_model_name(already) == "openrouter/z-ai/glm-5.2"
    codex = Config(agent={"provider": "codex"}, model={"model": "z-ai/glm-5.2"}, api={"provider": "openrouter"})
    assert hb._bench_model_name(codex) == "z-ai/glm-5.2"


def test_harbor_classify_remote_source():
    assert hb._classify_remote_source("terminal-bench@2.0", None, None) == ("registry", "terminal-bench@2.0")
    assert hb._classify_remote_source("terminal-bench", None, "2.0") == ("registry", "terminal-bench@2.0")
    assert hb._classify_remote_source("org/name", None, None) == ("package", "org/name@latest")
    assert hb._classify_remote_source("org/name@ref", None, None) == ("package", "org/name@ref")
    assert hb._classify_remote_source("ds", "https://github.com/o/r", None) == ("repo", "ds")


def test_harbor_image_names_computed_and_from_trial():
    import types as _t

    # No trial (error path): the deterministic sanitize("hb__" + task id), lowercased.
    assert hb._harbor_image_names(None, "Adaptive-Rejection-Sampler") == ["hb__adaptive-rejection-sampler"]
    # With a trial: prefer the actual env image name(s) AND include the computed one.
    trial = _t.SimpleNamespace(
        agent_environment=_t.SimpleNamespace(_main_image_name="hb__custom"),
        verifier_environment=None,
        task=_t.SimpleNamespace(short_name="my-task"),
    )
    names = hb._harbor_image_names(trial, "ignored")
    assert "hb__custom" in names and "hb__my-task" in names


def test_harbor_image_names_include_prebuilt_docker_image(tmp_path):
    """A task with ``[environment].docker_image`` runs in harbor's prebuilt mode: no hb__
    image is ever built, and harbor's ``down --rmi local`` teardown never removes a pulled
    tagged image — so the purge list must carry the task.toml name (with or without a trial)."""
    pytest.importorskip("tomllib")  # py3.11+; the harbor backend itself requires 3.12
    import types as _t

    task_dir = tmp_path / "fix-ocaml-gc"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        '[environment]\ndocker_image = "alexgshaw/fix-ocaml-gc:20251031"\n'
        '[verifier.environment]\ndocker_image = "alexgshaw/verifier:v1"\n',
        encoding="utf-8",
    )
    names = hb._harbor_image_names(None, "fix-ocaml-gc", task_dir)
    assert "alexgshaw/fix-ocaml-gc:20251031" in names
    assert "alexgshaw/verifier:v1" in names  # a separate verifier env's prebuilt image too
    assert "hb__fix-ocaml-gc" in names  # deterministic hb__ fallback still present

    trial = _t.SimpleNamespace(
        agent_environment=_t.SimpleNamespace(_main_image_name="hb__custom"),
        verifier_environment=None,
        task=_t.SimpleNamespace(short_name="fix-ocaml-gc"),
    )
    assert "alexgshaw/fix-ocaml-gc:20251031" in hb._harbor_image_names(trial, "fix-ocaml-gc", task_dir)


def test_harbor_image_names_tolerate_missing_or_bad_task_toml(tmp_path):
    assert hb._harbor_image_names(None, "t", tmp_path) == ["hb__t"]  # no task.toml
    (tmp_path / "task.toml").write_text("not = valid [ toml", encoding="utf-8")
    assert hb._harbor_image_names(None, "t", tmp_path) == ["hb__t"]  # unparseable -> best-effort
    assert hb._harbor_image_names(None, "t", None) == ["hb__t"]  # no task dir at all


def test_purge_images_invokes_docker_rmi(monkeypatch):
    calls = []
    monkeypatch.setattr(hb.subprocess, "run", lambda args, **kw: calls.append(args))
    hb._purge_images(["hb__a", "hb__b"])
    assert calls == [["docker", "rmi", "-f", "hb__a"], ["docker", "rmi", "-f", "hb__b"]]


def test_purge_images_swallows_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("no docker")

    monkeypatch.setattr(hb.subprocess, "run", boom)
    hb._purge_images(["hb__a"])  # best-effort: must not raise


def test_harbor_run_purges_image_unless_kept(monkeypatch, tmp_path):
    import types as _t

    fake_trial = _t.SimpleNamespace(
        agent_environment=_t.SimpleNamespace(_main_image_name="hb__t1"),
        verifier_environment=None,
        task=_t.SimpleNamespace(short_name="t1"),
    )
    fake_result = _t.SimpleNamespace(exception_info=None, verifier_result={"rewards": {"reward": 1.0}})
    monkeypatch.setattr(hb, "_build_trial_config", lambda *_a, **_k: object())
    monkeypatch.setattr(hb, "_create_and_run", lambda _config: "coro")
    monkeypatch.setattr(hb.asyncio, "run", lambda _coro: (fake_trial, fake_result))
    monkeypatch.setattr(hb, "_agent_dir", lambda _trial: None)
    purged: list[str] = []
    monkeypatch.setattr(hb, "_purge_images", purged.extend)
    src = BenchSource(type="harbor", source="S")

    hb.HarborBackend().run(Config(output={"traces_dir": tmp_path / "o"}), src, base.BenchTask(id="t1", raw=tmp_path))
    assert "hb__t1" in purged

    purged.clear()
    kept = Config(output={"traces_dir": tmp_path / "o2", "keep_bench_images": True})
    hb.HarborBackend().run(kept, src, base.BenchTask(id="t1", raw=tmp_path))
    assert purged == []  # keep_bench_images -> no purge


def test_harbor_run_purges_fallback_image_when_trial_fails(monkeypatch, tmp_path):
    """When _create_and_run blows up before returning a trial, the finally still purges the
    deterministic sanitize("hb__" + task_id) fallback so a failed setup can't leak its image."""
    monkeypatch.setattr(hb, "_build_trial_config", lambda *_a, **_k: object())

    def _explode(_config):
        raise RuntimeError("setup failed before any trial")

    # _create_and_run(config) raises while building asyncio.run's argument, so trial stays None.
    monkeypatch.setattr(hb, "_create_and_run", _explode)
    purged: list[str] = []
    monkeypatch.setattr(hb, "_purge_images", purged.extend)
    src = BenchSource(type="harbor", source="S")

    with pytest.raises(RuntimeError, match="setup failed"):
        hb.HarborBackend().run(
            Config(output={"traces_dir": tmp_path / "o"}), src, base.BenchTask(id="my_task", raw=tmp_path)
        )
    assert purged == ["hb__my_task"]  # trial=None -> deterministic fallback (underscore preserved)


def test_harbor_run_purges_prebuilt_image_on_interrupt(monkeypatch, tmp_path):
    """Ctrl-C mid-task (KeyboardInterrupt, no trial yet): the finally must still purge the
    prebuilt task image — terminal-bench tasks are all prebuilt, so leaving it leaks GBs."""
    pytest.importorskip("tomllib")
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('[environment]\ndocker_image = "example/t1:v1"\n', encoding="utf-8")
    monkeypatch.setattr(hb, "_build_trial_config", lambda *_a, **_k: object())

    def _interrupt(_config):
        raise KeyboardInterrupt

    monkeypatch.setattr(hb, "_create_and_run", _interrupt)
    purged: list[str] = []
    monkeypatch.setattr(hb, "_purge_images", purged.extend)

    with pytest.raises(KeyboardInterrupt):
        hb.HarborBackend().run(
            Config(output={"traces_dir": tmp_path / "o"}),
            BenchSource(type="harbor", source="S"),
            base.BenchTask(id="t1", raw=task_dir),
        )
    assert "example/t1:v1" in purged
    assert "hb__t1" in purged


def test_harbor_run_keeps_prebuilt_image_shared_across_tasks(monkeypatch, tmp_path):
    """A prebuilt image declared by more than one task in the source must NOT be purged
    per-task: rmi'ing it can race a concurrent sibling's `compose up` (No such image) and
    forces a re-pull per task. Only images unique to the task are removed."""
    pytest.importorskip("tomllib")
    root = tmp_path / "tasks"
    for name, image in (("t1", "example/shared:v1"), ("t2", "example/shared:v1"), ("t3", "example/solo:v1")):
        d = root / name
        d.mkdir(parents=True)
        (d / "task.toml").write_text(f'[environment]\ndocker_image = "{image}"\n', encoding="utf-8")
    monkeypatch.setattr(hb, "_build_trial_config", lambda *_a, **_k: object())

    def _explode(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(hb, "_create_and_run", _explode)
    purged: list[str] = []
    monkeypatch.setattr(hb, "_purge_images", purged.extend)

    cfg = Config(output={"traces_dir": tmp_path / "o"})
    src = BenchSource(type="harbor", source=str(root))
    backend = hb.HarborBackend()
    tasks = {t.id: t for t in backend.tasks(cfg, src)}

    with pytest.raises(RuntimeError):
        backend.run(cfg, src, tasks["t1"])
    assert "example/shared:v1" not in purged  # t2 declares the same image
    assert "hb__t1" in purged  # per-task hb__ name still purged

    purged.clear()
    with pytest.raises(RuntimeError):
        backend.run(cfg, src, tasks["t3"])
    assert "example/solo:v1" in purged  # unique to t3 -> purged


def test_harbor_run_purges_image_duplicated_within_one_task(monkeypatch, tmp_path):
    """One task declaring the same image for both its agent and verifier environments is
    NOT sharing across tasks: the image is still unique to the task and must be purged."""
    pytest.importorskip("tomllib")
    root = tmp_path / "tasks"
    d1, d2 = root / "t1", root / "t2"
    d1.mkdir(parents=True)
    (d1 / "task.toml").write_text(
        '[environment]\ndocker_image = "example/both:v1"\n'
        '[verifier.environment]\ndocker_image = "example/both:v1"\n',
        encoding="utf-8",
    )
    d2.mkdir()
    (d2 / "task.toml").write_text('[environment]\ndocker_image = "example/other:v1"\n', encoding="utf-8")
    monkeypatch.setattr(hb, "_build_trial_config", lambda *_a, **_k: object())

    def _explode(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(hb, "_create_and_run", _explode)
    purged: list[str] = []
    monkeypatch.setattr(hb, "_purge_images", purged.extend)

    cfg = Config(output={"traces_dir": tmp_path / "o"})
    src = BenchSource(type="harbor", source=str(root))
    backend = hb.HarborBackend()
    tasks = {t.id: t for t in backend.tasks(cfg, src)}

    with pytest.raises(RuntimeError):
        backend.run(cfg, src, tasks["t1"])
    assert "example/both:v1" in purged  # duplicated within t1 only -> still unique -> purged


def test_harbor_resolve_task_dirs(tmp_path):
    single = tmp_path / "one"
    single.mkdir()
    (single / "task.toml").write_text("", encoding="utf-8")
    assert hb._resolve_task_dirs(single) == [single]
    coll = tmp_path / "many"
    (coll / "a").mkdir(parents=True)
    (coll / "a" / "task.toml").write_text("", encoding="utf-8")
    (coll / "b").mkdir()
    (coll / "b" / "task.toml").write_text("", encoding="utf-8")
    assert hb._resolve_task_dirs(coll) == [coll / "a", coll / "b"]
    with pytest.raises(RuntimeError, match="No Harbor tasks"):
        empty = tmp_path / "empty"
        empty.mkdir()
        hb._resolve_task_dirs(empty)


# A minimal pi `--mode json` stream (as harbor's --no-session pi run emits to pi.txt).
_PI_STREAM_LINES = [
    'Warning: Model "z-ai/glm-5.2" not found for provider "openrouter". Using custom model id.',
    json.dumps({"type": "session", "version": 3, "id": "abc", "cwd": "/app"}),
    json.dumps({"type": "agent_start"}),
    json.dumps({"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "Fix add()"}]}}),
    json.dumps({"type": "message_end", "message": {"role": "assistant", "provider": "openrouter",
        "model": "z-ai/glm-5.2", "content": [{"type": "text", "text": "Fixed."}]}}),
]


def test_pi_stream_to_session_events(tmp_path):
    pi_txt = tmp_path / "pi.txt"
    pi_txt.write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    events = hb._pi_stream_to_session_events(pi_txt)
    assert [e["type"] for e in events] == ["session", "message", "model_change", "message"]
    mc = next(e for e in events if e["type"] == "model_change")
    assert mc == {"type": "model_change", "provider": "openrouter", "modelId": "z-ai/glm-5.2"}


def test_native_trace_pi_stream(tmp_path):
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source="S")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")
    lines, native_dir = hb._native_trace(cfg, source, agent_dir, "add-bug")
    # session dir is namespaced by source so concurrent same-named tasks can't race
    assert lines and native_dir == tmp_path / "bench" / "sessions" / base.bench_stem(source, "add-bug")
    assert (native_dir / "pi.jsonl").is_file()
    types = {json.loads(line)["type"] for line in lines}
    assert "session" in types and "message" in types


def test_rewards_from_result_and_files(tmp_path):
    class R:
        verifier_result = {"rewards": {"reward": 1.0, "sub": 0.5}}
    assert hb._rewards_from_result(R()) == {"reward": 1.0, "sub": 0.5}

    class N:
        verifier_result = None
    assert hb._rewards_from_result(N()) is None

    d = tmp_path / "trial"
    d.mkdir()
    (d / "reward.txt").write_text("0.0\n", encoding="utf-8")
    assert hb._rewards_from_files(d) == {"reward": 0.0}


# ------------------------------------------------------------------- harbor backend run/tasks

def test_harbor_tasks_local(tmp_path):
    pytest.importorskip("harbor")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "add-bug").mkdir(parents=True)
    (tasks_dir / "add-bug" / "task.toml").write_text("", encoding="utf-8")
    cfg = Config(output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source=str(tasks_dir))
    tasks = list(hb.HarborBackend().tasks(cfg, source))
    assert [t.id for t in tasks] == ["add-bug"]


def test_harbor_run_builds_benchrun_from_trial(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "pi.txt").write_text("\n".join(_PI_STREAM_LINES) + "\n", encoding="utf-8")

    class _Trial:
        paths = type("P", (), {"agent_dir": agent_dir})()

    class _Result:
        verifier_result = {"rewards": {"reward": 1.0}}
        exception_info = None

    async def _fake_create_and_run(config):
        return _Trial(), _Result()

    monkeypatch.setattr(hb, "_create_and_run", _fake_create_and_run)
    monkeypatch.setattr(hb, "_build_trial_config", lambda cfg, source, task_dir, trials_dir: object())
    cfg = Config(agent={"provider": "pi"}, output={"traces_dir": tmp_path / "output"})
    source = BenchSource(type="harbor", source="terminal-bench@2.0")
    run = hb.HarborBackend().run(cfg, source, base.BenchTask(id="add-bug", raw=tmp_path / "task"))
    assert run.rewards == {"reward": 1.0}
    assert run.native_lines and run.metadata.get("exception") is None


# ------------------------------------------------------------------- driver

class _FakeBackend:
    type = "fake"

    def __init__(self, runs):
        self.runs = runs
        self.ran: list[str] = []

    def require(self):
        pass

    def tasks(self, cfg, source, *, refresh=False):
        return [base.BenchTask(id=task_id) for task_id in self.runs]

    def run(self, cfg, source, task):
        self.ran.append(task.id)
        result = self.runs[task.id]
        if isinstance(result, Exception):
            raise result
        return result


def _bench_cfg(tmp_path):
    return Config(
        agent={"provider": "pi"},
        bench={"sources": [{"type": "fake", "source": "S"}]},
        output={"traces_dir": tmp_path / "output"},
    )


def test_run_bench_drives_sources_and_harvests(tmp_path, monkeypatch):
    fake = _FakeBackend({
        "t1": base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0}),
        "t2": base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 0.0}),
    })
    monkeypatch.setattr(bench_runner, "get_backend", lambda t: fake)
    written = run_bench(_bench_cfg(tmp_path))
    assert len(written) == 2 and fake.ran == ["t1", "t2"]
    s = BenchSource(type="fake", source="S")
    assert (tmp_path / "output" / "passed" / f"{base.bench_stem(s, 't1')}.jsonl").is_file()
    assert (tmp_path / "output" / "failed" / f"{base.bench_stem(s, 't2')}.jsonl").is_file()


def test_run_bench_raises_when_all_tasks_fail(tmp_path, monkeypatch):
    # Per-task errors are swallowed; if EVERY dispatched task fails, run_bench must raise (so the
    # CLI exits non-zero) rather than return empty and look like a successful empty benchmark.
    fake = _FakeBackend({"t1": RuntimeError("docker boom"), "t2": RuntimeError("docker boom")})
    monkeypatch.setattr(bench_runner, "get_backend", lambda t: fake)
    with pytest.raises(RuntimeError, match="all 2 attempted task"):
        run_bench(_bench_cfg(tmp_path))


def test_run_bench_resume_skips_and_failure_continues(tmp_path, monkeypatch):
    s = BenchSource(type="fake", source="S")
    # Pre-harvest t1 so resume skips it; t2 raises (skipped); t3 succeeds.
    done = tmp_path / "output" / "passed" / f"{base.bench_stem(s, 't1')}.jsonl"
    done.parent.mkdir(parents=True)
    done.write_text('{"type":"session"}\n', encoding="utf-8")
    fake = _FakeBackend({
        "t1": base.BenchRun(native_lines=['{"x":1}'], rewards={"reward": 1.0}),
        "t2": RuntimeError("docker boom"),
        "t3": base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0}),
    })
    monkeypatch.setattr(bench_runner, "get_backend", lambda t: fake)
    run_bench(_bench_cfg(tmp_path), resume=True)
    assert "t1" not in fake.ran  # skipped via resume
    assert fake.ran == ["t2", "t3"]
    assert (tmp_path / "output" / "passed" / f"{base.bench_stem(s, 't3')}.jsonl").is_file()


def test_run_bench_unknown_type_aborts(tmp_path):
    # An unregistered source type -> a clear unknown-type error (harbor + swe-bench are known).
    cfg = Config(bench={"sources": [{"type": "nope", "source": "x"}]}, output={"traces_dir": tmp_path / "o"})
    with pytest.raises(RuntimeError, match="Unknown bench source type"):
        run_bench(cfg)


def test_run_bench_honors_max_concurrency(tmp_path, monkeypatch):
    import threading

    max_workers = 3
    # A barrier makes "the pool parallelizes up to the cap" deterministic instead of
    # sleep-timing-dependent: exactly max_workers tasks must be in-flight to release each wave.
    barrier = threading.Barrier(max_workers, timeout=10)
    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    class _ConcBackend:
        type = "fake"

        def require(self):
            pass

        def tasks(self, cfg, source, *, refresh=False):
            return [base.BenchTask(id=f"t{i}") for i in range(6)]

        def run(self, cfg, source, task):
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            try:
                barrier.wait()  # blocks until max_workers are concurrently in-flight
            except threading.BrokenBarrierError:
                pass
            with lock:
                state["current"] -= 1
            return base.BenchRun(native_lines=['{"type":"session"}'], rewards={"reward": 1.0})

    monkeypatch.setattr(bench_runner, "get_backend", lambda t: _ConcBackend())
    cfg = Config(
        agent={"provider": "pi"},
        bench={"sources": [{"type": "fake", "source": "S"}]},
        output={"traces_dir": tmp_path / "output"},
        max_concurrency=max_workers,
    )
    written = run_bench(cfg)
    assert len(written) == 6
    assert state["peak"] == max_workers  # pool parallelizes exactly up to the cap

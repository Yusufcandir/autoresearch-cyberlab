"""AT-16 to AT-19: typed plugin fixtures, exact policy evaluation, metadata without execution,
controlled code amendment."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from cyberlab_schemas.plugins import (
    ISOLATION_REQUIRED,
    METADATA_REQUIRES_EXECUTION,
    PATH_NOT_PERMITTED,
    SCHEMA_INVALID,
    UNAVAILABLE_FEATURE_VIEW,
    PluginError,
    ResearchAdapter,
    list_capabilities,
    load_adapter,
)

# ---------------------------------------------------------------- AT-16 typed fixtures


def test_every_plugin_implements_the_adapter_contract_at16() -> None:
    names = {c.name for c in list_capabilities()}
    assert {"text", "audio", "graph_policy", "code_artifact"} <= names
    for name in ("text", "audio", "graph_policy", "code_artifact"):
        assert isinstance(load_adapter(name), ResearchAdapter)
    with pytest.raises(PluginError) as exc:
        load_adapter("video")
    assert exc.value.code == UNAVAILABLE_FEATURE_VIEW


def test_text_fixture_validates_and_rejects_bad_schema_at16() -> None:
    from cyberlab_tabular.text import TextAdapter, fixture

    a = TextAdapter()
    report = a.validate_dataset(fixture())
    assert report["ok"] and report["stats"]["languages"]["en"] == 4
    split = a.feature_view("group_split", fixture())
    assert not set(split["train"]) & set(split["test"])
    assert len(a.feature_view("hashed_bow", fixture())["rows"]["d1"]) == 64
    bad = fixture()
    bad["documents"][0]["language"] = "english"
    with pytest.raises(PluginError) as exc:
        a.validate_dataset(bad)
    assert exc.value.code == SCHEMA_INVALID
    with pytest.raises(PluginError) as exc:
        a.feature_view("sentence_embeddings", fixture())
    assert exc.value.code == UNAVAILABLE_FEATURE_VIEW


def test_audio_fixture_validates_rate_and_offers_no_transcript_at16(tmp_path: Path) -> None:
    from cyberlab_audio import AudioAdapter, write_fixture

    a = AudioAdapter()
    manifest = write_fixture(tmp_path / "audio")
    report = a.validate_dataset(manifest)
    assert report["ok"] and report["groups"] == ["spk1", "spk2", "spk3"]
    feats = a.feature_view("acoustic_summary", manifest)["rows"]
    assert (
        feats["c0"]["spectral_centroid_hz"] > feats["c1"]["spectral_centroid_hz"]
    )  # 880 vs 220 Hz
    split = a.feature_view("speaker_split", manifest)
    assert not set(split["train"]) & set(split["test"])
    with wave.open(str(tmp_path / "audio" / "clip0.wav"), "wb") as w:  # wrong sample rate
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 100)
    bad = a.validate_dataset(manifest)
    assert not bad["ok"] and "16000 Hz, expected 8000 Hz" in bad["problems"][0]
    with pytest.raises(PluginError) as exc:
        a.feature_view("transcript", manifest)
    assert exc.value.code == UNAVAILABLE_FEATURE_VIEW
    escape = {**manifest, "clips": [{**manifest["clips"][0], "path": "../../secret.wav"}] * 2}
    with pytest.raises(PluginError):
        a.validate_dataset(escape)


def test_graph_fixture_validates_structure_at16() -> None:
    from cyberlab_graph import GraphAdapter, fixture

    g = GraphAdapter()
    assert g.validate_dataset(fixture())["ok"]
    split = g.feature_view("component_split", fixture())
    assert not set(split["train"]) & set(split["test"])
    broken = fixture()
    broken["edges"].append({"src": "user:alice", "dst": "role:ghost", "type": "member_of"})
    report = g.validate_dataset(broken)
    assert not report["ok"] and "unknown vertices" in report["problems"][0]


def test_code_fixture_is_analysed_statically_at16(tmp_path: Path) -> None:
    from cyberlab_code import CodeAdapter, snapshot_manifest, write_fixture

    root = write_fixture(tmp_path / "proj")
    c = CodeAdapter()
    manifest = snapshot_manifest(root)
    assert c.validate_dataset(manifest)["ok"]
    rows = c.feature_view("static_summary", manifest)["rows"]
    assert rows["src/calc.py"]["functions"] == 1 and rows["src/calc.py"]["security_findings"] == []
    (root / "src" / "calc.py").write_text("x = 1\n")
    assert "hash does not match" in c.validate_dataset(manifest)["problems"][0]


# ---------------------------------------------------------------- AT-17 exact policy evaluator

POLICY = {
    "version": "1",
    "statements": [
        {
            "sid": "readers-read",
            "effect": "allow",
            "principals": ["role:reader"],
            "actions": ["read"],
            "resources": ["dataset/*"],
        },
        {
            "sid": "no-contractors-private",
            "effect": "deny",
            "principals": ["group:contractors"],
            "actions": ["*"],
            "resources": ["dataset/private-*"],
        },
        {
            "sid": "admins-write-office",
            "effect": "allow",
            "principals": ["role:admin"],
            "actions": ["write"],
            "resources": ["dataset/*"],
            "conditions": {"network": "office"},
        },
    ],
}


def test_exact_evaluator_decides_and_model_score_cannot_override_deny_at17() -> None:
    from cyberlab_graph import decide, evaluate, fixture

    graph = fixture()
    # membership resolved through the graph: alice -> analysts -> reader
    ok = evaluate(
        POLICY, {"principal": "user:alice", "action": "read", "resource": "dataset/cic"}, graph
    )
    assert ok.effect == "allow" and "role:reader" in ok.identities
    # explicit deny beats the inherited allow
    req = {"principal": "user:mallory", "action": "read", "resource": "dataset/private-hr"}
    d = decide(POLICY, req, graph, model_score=0.01)  # the model is confident this is safe
    assert d.effect == "deny" and d.matched_deny == ["no-contractors-private"]
    assert d.model_advice == "safe" and not d.flagged_for_review
    # default deny: nothing allows bob to write
    w = decide(
        POLICY,
        {"principal": "user:bob", "action": "write", "resource": "dataset/cic"},
        graph,
        model_score=0.0,
    )
    assert w.effect == "deny" and "default deny" in w.reason
    # a risky score on an allowed request only flags it for review
    r = decide(
        POLICY,
        {"principal": "user:alice", "action": "read", "resource": "dataset/cic"},
        graph,
        model_score=0.97,
    )
    assert r.effect == "allow" and r.flagged_for_review


def test_invalid_policy_is_a_typed_error_at17() -> None:
    from cyberlab_graph import evaluate

    with pytest.raises(PluginError) as exc:
        evaluate(
            {
                "version": "1",
                "statements": [
                    {
                        "sid": "x",
                        "effect": "maybe",
                        "principals": ["*"],
                        "actions": ["*"],
                        "resources": ["*"],
                    }
                ],
            },
            {"principal": "a", "action": "b", "resource": "c"},
        )
    assert exc.value.code == "POLICY_INVALID"


# ---------------------------------------------------------------- AT-18 metadata without execution


def test_package_metadata_is_read_without_running_install_scripts_at18(tmp_path: Path) -> None:
    from cyberlab_code import extract_metadata
    from cyberlab_code.metadata import make_sdist_fixture

    canary = tmp_path / "EXECUTED"
    sdist = make_sdist_fixture(tmp_path / "demo-1.0.tar.gz", canary=canary)
    meta = extract_metadata(sdist)
    assert meta["name"] == "demo" and meta["version"] == "1.0" and meta["license"] == "MIT"
    assert meta["dependencies"] == ["requests>=2"] and meta["executed"] is False
    assert any(s.endswith("setup.py") for s in meta["install_scripts_present"])
    assert not canary.exists()  # setup.py never ran

    only_setup = make_sdist_fixture(
        tmp_path / "bare-1.0.tar.gz", with_pkg_info=False, canary=canary
    )
    with pytest.raises(PluginError) as exc:
        extract_metadata(only_setup)
    assert exc.value.code == METADATA_REQUIRES_EXECUTION
    assert not canary.exists()


# ---------------------------------------------------------------- AT-19 controlled amendment

FIX = """--- a/src/calc.py
+++ b/src/calc.py
@@ -1,2 +1,2 @@
 def mean(values):
-    return sum(values) / (len(values) + 1)
+    return sum(values) / len(values)
"""


def test_amendment_touches_only_a_copy_and_links_receipts_at19(tmp_path: Path) -> None:
    from cyberlab_code import amend, write_fixture
    from cyberlab_code.amend import tree_sha256

    snap = write_fixture(tmp_path / "snap")
    before = tree_sha256(snap)
    work = tmp_path / "work"
    work.mkdir()

    receipt = amend(snap, FIX, ["src/*.py"], work)  # no isolated runner configured
    assert tree_sha256(snap) == before  # the snapshot itself is never written
    assert list(receipt["changed_files"]) == ["src/calc.py"]
    assert receipt["snapshot_sha256"] == before != receipt["amended_snapshot_sha256"]
    assert receipt["security_check"]["passed"]
    assert receipt["functional_tests"]["status"] == ISOLATION_REQUIRED
    assert receipt["status"] == "blocked"
    fixed = Path(receipt["amended_path"]) / "src" / "calc.py"
    assert "/ len(values)\n" in fixed.read_text()

    calls = []

    def isolated_runner(path: Path) -> dict:
        calls.append(path)
        return {"backend": "test-double", "passed": True}

    ok = amend(snap, FIX, ["src/*.py"], work, test_runner=isolated_runner)
    assert ok["status"] == "accepted" and calls == [Path(ok["amended_path"])]
    assert len(ok["functional_tests"]["receipt_sha256"]) == 64
    assert (work / f"amendment-{ok['amendment_id']}.json").exists()


@pytest.mark.parametrize(
    ("target", "why"),
    [
        ("tests/test_calc.py", "test files"),
        ("../outside.py", "unsafe path"),
        ("setup.cfg", "outside the permitted"),
    ],
)
def test_amendment_outside_the_permitted_paths_is_refused_at19(
    tmp_path: Path, target: str, why: str
) -> None:
    from cyberlab_code import amend, write_fixture

    snap = write_fixture(tmp_path / "snap")
    diff = f"--- a/{target}\n+++ b/{target}\n@@ -1,1 +1,1 @@\n-x\n+y\n"
    with pytest.raises(PluginError) as exc:
        amend(
            snap,
            diff,
            ["src/*.py", "tests/*.py", "*.cfg"] if why == "test files" else ["src/*.py"],
            tmp_path,
        )
    assert exc.value.code == PATH_NOT_PERMITTED and why in exc.value.message


def test_amendment_with_dangerous_code_is_rejected_before_any_test_at19(tmp_path: Path) -> None:
    from cyberlab_code import amend, write_fixture

    snap = write_fixture(tmp_path / "snap")
    evil = FIX.replace(
        "+    return sum(values) / len(values)",
        "+    import subprocess; subprocess.run(['curl', 'x'])\n+    return sum(values) / len(values)",
    ).replace("@@ -1,2 +1,2 @@", "@@ -1,2 +1,3 @@")
    ran = []
    receipt = amend(
        snap, evil, ["src/*.py"], tmp_path, test_runner=lambda p: ran.append(p) or {"passed": True}
    )
    assert receipt["status"] == "rejected" and not ran
    assert any("subprocess" in f for f in receipt["security_check"]["findings"])

"""Guards over report.schema.json, the published contract of the JSON report.

The schema is documentation, not something the exporter reads, so nothing makes
it true on its own: the day a field is added to generate_json_report(), only
these tests notice that the schema still describes the old payload.
"""
from __future__ import annotations

import importlib.resources
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv.models import (
    REASONING_SCOPE_ALL,
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    ReasoningStats,
    ReasoningUnit,
    TestCaseResult,
)
from prompttestenv.json_report import generate_json_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "report.schema.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# ── A minimal structural checker ──────────────────────────────────────────────
# Deliberately not `jsonschema`: the one property worth enforcing here is that
# the schema and the payload describe the same set of keys, and a whole
# validation dependency for that would be more machinery than the guard needs.


class SchemaWalker:
    """Walks a payload against a schema, collecting key-level mismatches."""

    def __init__(self, schema: dict):
        self.schema = schema
        self.problems: list[str] = []

    def _resolve(self, node: dict) -> dict:
        """Inline a local $ref and flatten allOf into one node."""
        merged: dict = {}
        for part in [node] + list(node.get("allOf", [])):
            if "$ref" in part:
                target = self.schema
                for step in part["$ref"].lstrip("#/").split("/"):
                    target = target[step]
                part = self._resolve(target)
            for key, value in part.items():
                if key in ("properties", "$defs"):
                    merged.setdefault(key, {}).update(value)
                elif key == "required":
                    merged.setdefault("required", [])
                    merged["required"] = list(merged["required"]) + list(value)
                elif key != "allOf":
                    merged.setdefault(key, value)
        return merged

    def check(self, payload, node: dict, path: str = "$") -> None:
        node = self._resolve(node)

        # Before the type branches: a oneOf node carries no properties of its
        # own, so handling it later would let every object under a nullable
        # field through unchecked.
        if "oneOf" in node:
            if payload is None:
                return
            for option in node["oneOf"]:
                if self._resolve(option).get("type") != "null":
                    self.check(payload, option, path)
            return

        if isinstance(payload, dict):
            props = node.get("properties", {})
            # unevaluatedProperties is how draft 2020-12 closes an object
            # composed with allOf; additionalProperties cannot see the
            # properties a $ref contributed, so the composed nodes use it.
            extra = node.get("additionalProperties", node.get("unevaluatedProperties", True))
            for key in node.get("required", []):
                if key not in payload:
                    self.problems.append(f"{path}: schema requires '{key}', payload has no such key")
            for key, value in payload.items():
                if key in props:
                    self.check(value, props[key], f"{path}.{key}")
                elif isinstance(extra, dict):
                    self.check(value, extra, f"{path}.{key}")
                elif extra is False:
                    self.problems.append(f"{path}: payload key '{key}' is not declared in the schema")
            return

        if isinstance(payload, list) and "items" in node:
            for index, item in enumerate(payload):
                self.check(item, node["items"], f"{path}[{index}]")
            return



def build_payload(project_dir: str, *, grouped: bool, filename: str) -> dict:
    """Render a JSON report exercising every branch of the exporter."""
    candidates = [
        Candidate(
            name="Baseline", provider="google", model="gemini",
            system_prompt_file=None, resolved_system_instruction="RESOLVED SECRET",
        ),
        Candidate(
            name="Pirate", provider="ollama", model="gemma", thinking=True,
            system_prompt_file="pirate.txt", resolved_system_instruction="Arr.",
        ),
    ]

    analysed = ReasoningStats(
        units=[ReasoningUnit(start=0, end=5, framing=3.0), ReasoningUnit(start=6, end=11, solving=2.0)],
        coverage_framing=0.5, coverage_solving=0.5, coverage_presentation=0.0,
        density=1.0, alt_path=1, autocorrect=0, alignment_score=9,
        alt_path_units=[1], autocorrect_units=[], alignment_evidence=[],
        repetition_rate=0.0, trace_response_drift=0.42,
        reasoning_is_summary=True, schema_stamp="framing+solving+presentation@deadbeef",
    )

    rows = []
    for test_id, group in (("t1", "Writing"), ("t2", "Extraction")):
        row = TestCaseResult(
            test_id=test_id, prompt="p", criteria="c", group=group,
            files_used=["test_files/sample.txt"],
        )
        for name, analysis in (("Baseline", analysed), ("Pirate", None)):
            perf = CandidatePerformance()
            perf.scores.extend([8.0, -1.0])
            perf.global_scores.extend([6.0, 6.0])
            perf.times.extend([1.25, 2.5])
            perf.tokens.extend([150, 160])
            perf.reasoning_tokens.extend([200, 210])
            perf.best_output = "hello"
            perf.best_reason = "good"
            perf.best_global_reason = "fine"
            if analysis is not None:
                perf.best_reasoning_text = "first second"
                perf.best_reasoning_analysis = analysis
                perf.reasoning_analyses.append(analysis.to_dict())
            row.candidates_perf[name] = perf
        rows.append(row)

    verdict = "Plain verdict text."
    if grouped:
        verdict = json.dumps({
            "is_grouped": True,
            "groups": [{"group_name": "Writing", "verdict": "Group body."}],
            "global_verdict": "Overall body.",
        })

    judge_config = JudgeConfig()
    judge_config.reasoning_analysis = REASONING_SCOPE_ALL
    path = generate_json_report(
        project_dir, rows, candidates, verdict,
        GlobalCriteria(mode="llm-judge", llm_judge_criteria="Be concise."),
        judge_config, filename=filename,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TestSchemaCopiesAgree(unittest.TestCase):
    def test_shipped_copy_matches_the_repo_root_schema(self):
        """Two copies exist so a wheel install still ships one; they must not drift apart."""
        packaged = json.loads(
            importlib.resources.files("prompttestenv")
            .joinpath("templates/report.schema.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(load_schema(), packaged)


class TestSchemaDescribesTheRealPayload(unittest.TestCase):
    """The guard that keeps report.schema.json honest as the exporter changes."""

    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_schema_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.schema = load_schema()

    def _assert_agrees(self, payload: dict) -> None:
        walker = SchemaWalker(self.schema)
        walker.check(payload, self.schema)
        self.assertEqual(
            walker.problems, [],
            "report.schema.json and generate_json_report() disagree:\n  "
            + "\n  ".join(walker.problems),
        )

    def test_plain_verdict_payload_matches_the_schema(self):
        self._assert_agrees(build_payload(self.project_dir, grouped=False, filename="plain.json"))

    def test_grouped_verdict_payload_matches_the_schema(self):
        self._assert_agrees(build_payload(self.project_dir, grouped=True, filename="grouped.json"))

    def test_declared_schema_version_matches_the_exporter(self):
        payload = build_payload(self.project_dir, grouped=False, filename="version.json")
        self.assertEqual(payload["schema_version"], "1.0")

    def test_the_guard_actually_catches_an_undeclared_key(self):
        """Without this, a walker that silently passes everything looks like a green guard."""
        payload = build_payload(self.project_dir, grouped=False, filename="canary.json")
        payload["surprise"] = 1
        walker = SchemaWalker(self.schema)
        walker.check(payload, self.schema)
        self.assertTrue(
            any("surprise" in problem for problem in walker.problems),
            "the walker did not notice an undeclared top-level key",
        )

    def test_the_guard_actually_catches_a_key_under_a_composed_object(self):
        """The blind spot the first version of this walker had.

        The per-candidate records compose $defs/performance through allOf, and a
        node that only inherits its properties by reference is exactly where a
        naive walker stops descending.
        """
        payload = build_payload(self.project_dir, grouped=False, filename="canary3.json")
        payload["aggregate"]["Baseline"]["surprise"] = 1
        payload["test_cases"][0]["candidates"]["Baseline"]["best"]["surprise"] = 1
        walker = SchemaWalker(self.schema)
        walker.check(payload, self.schema)
        self.assertEqual(
            len([p for p in walker.problems if "surprise" in p]), 2,
            f"the walker did not descend into both composed objects: {walker.problems}",
        )

    def test_the_guard_actually_catches_a_key_under_a_nullable_object(self):
        """reasoning_analysis is declared with oneOf, which hides properties one level down."""
        payload = build_payload(self.project_dir, grouped=False, filename="canary4.json")
        payload["test_cases"][0]["candidates"]["Baseline"]["best"]["reasoning_analysis"]["surprise"] = 1
        walker = SchemaWalker(self.schema)
        walker.check(payload, self.schema)
        self.assertTrue(
            any("surprise" in problem for problem in walker.problems),
            "the walker did not look inside a nullable object",
        )

    def test_the_guard_actually_catches_a_missing_required_key(self):
        payload = build_payload(self.project_dir, grouped=False, filename="canary2.json")
        del payload["aggregate"]
        walker = SchemaWalker(self.schema)
        walker.check(payload, self.schema)
        self.assertTrue(
            any("aggregate" in problem for problem in walker.problems),
            "the walker did not notice a missing required key",
        )


if __name__ == "__main__":
    unittest.main()

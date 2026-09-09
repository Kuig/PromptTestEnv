"""Tests for the reasoning-analysis phase, mainly its scope selection.

The scope decides how many judge calls a run buys: "all" analyses every
repetition, "best" only the highest-scoring one per candidate x test, which is
`repetitions` times cheaper and is the repetition the report draws anyway.
"""
import unittest
from unittest.mock import patch

from prompttestenv.analysis import attach_best, attach_reasoning, keys_to_analyze, run_analysis_phase
from prompttestenv.models import (
    REASONING_SCOPE_ALL,
    REASONING_SCOPE_BEST,
    REASONING_SCOPE_NONE,
    CandidatePerformance,
    JudgeConfig,
    ReasoningStats,
    TestCaseResult,
)
from prompttestenv.progress import ProgressState


def _state(gen: dict, evals: dict = None, reasoning: dict = None) -> ProgressState:
    """Build a ProgressState holding just the event indexes the phase reads."""
    return ProgressState(
        gen_events=gen,
        eval_events=evals or {},
        reasoning_events=reasoning or {},
    )


def _gen(**overrides) -> dict:
    event = {"reasoning_text": "A thought. Another thought.", "output": "answer"}
    event.update(overrides)
    return event


class TestKeysToAnalyze(unittest.TestCase):
    """Which traces a given scope spends judge calls on."""

    def setUp(self):
        # Two repetitions of one test, plus a second test, for one candidate.
        self.gen = {
            ("Alpha", "t1", 0): _gen(),
            ("Alpha", "t1", 1): _gen(),
            ("Alpha", "t2", 0): _gen(),
        }
        self.evals = {
            ("Alpha", "t1", 0): {"score": 4},
            ("Alpha", "t1", 1): {"score": 9},
            ("Alpha", "t2", 0): {"score": 7},
        }

    def test_all_scope_covers_every_repetition(self):
        keys = keys_to_analyze(_state(self.gen, self.evals), REASONING_SCOPE_ALL, False)
        self.assertEqual(
            keys, [("Alpha", "t1", 0), ("Alpha", "t1", 1), ("Alpha", "t2", 0)]
        )

    def test_best_scope_keeps_one_repetition_per_test(self):
        keys = keys_to_analyze(_state(self.gen, self.evals), REASONING_SCOPE_BEST, False)
        self.assertEqual(keys, [("Alpha", "t1", 1), ("Alpha", "t2", 0)])

    def test_best_scope_is_what_makes_it_cheaper(self):
        """The whole point: cost drops by the repetition count, not by a constant."""
        state = _state(self.gen, self.evals)
        self.assertLess(
            len(keys_to_analyze(state, REASONING_SCOPE_BEST, False)),
            len(keys_to_analyze(state, REASONING_SCOPE_ALL, False)),
        )

    def test_best_scope_falls_back_to_the_earliest_repetition_without_evals(self):
        """A run analysed before evaluation must still pick deterministically."""
        keys = keys_to_analyze(_state(self.gen), REASONING_SCOPE_BEST, False)
        self.assertEqual(keys, [("Alpha", "t1", 0), ("Alpha", "t2", 0)])

    def test_best_scope_breaks_ties_on_the_earliest_repetition(self):
        evals = {k: {"score": 8} for k in self.gen}
        keys = keys_to_analyze(_state(self.gen, evals), REASONING_SCOPE_BEST, False)
        self.assertEqual(keys, [("Alpha", "t1", 0), ("Alpha", "t2", 0)])

    def test_traces_that_are_empty_or_absent_are_never_analyzed(self):
        gen = {
            ("Alpha", "t1", 0): _gen(reasoning_text=""),
            ("Alpha", "t1", 1): _gen(reasoning_text="   \n "),
            ("Alpha", "t1", 2): _gen(),
        }
        del gen[("Alpha", "t1", 2)]["reasoning_text"]
        gen[("Alpha", "t1", 2)]["reasoning_text"] = None
        gen[("Alpha", "t2", 0)] = _gen()
        for scope in (REASONING_SCOPE_ALL, REASONING_SCOPE_BEST):
            with self.subTest(scope=scope):
                self.assertEqual(
                    keys_to_analyze(_state(gen), scope, False), [("Alpha", "t2", 0)]
                )

    def test_existing_analyses_are_skipped_unless_forced(self):
        state = _state(self.gen, self.evals, {("Alpha", "t1", 1): {"type": "reasoning"}})
        self.assertEqual(
            keys_to_analyze(state, REASONING_SCOPE_BEST, False), [("Alpha", "t2", 0)]
        )
        self.assertEqual(
            keys_to_analyze(state, REASONING_SCOPE_BEST, True),
            [("Alpha", "t1", 1), ("Alpha", "t2", 0)],
        )

    def test_force_keys_reanalyzes_just_those_keys(self):
        """A regenerated trace invalidates its analysis: the unit offsets
        would otherwise resolve into text that no longer exists."""
        state = _state(
            self.gen, self.evals,
            {("Alpha", "t1", 1): {"type": "reasoning"}, ("Alpha", "t2", 0): {"type": "reasoning"}},
        )
        self.assertEqual(keys_to_analyze(state, REASONING_SCOPE_BEST, False), [])
        self.assertEqual(
            keys_to_analyze(
                state, REASONING_SCOPE_BEST, False, frozenset({("Alpha", "t1", 1)})
            ),
            [("Alpha", "t1", 1)],
        )


class TestScopeIsResumable(unittest.TestCase):
    """Changing scope must never discard analyses already paid for."""

    def setUp(self):
        self.gen = {
            ("Alpha", "t1", 0): _gen(),
            ("Alpha", "t1", 1): _gen(),
        }
        self.evals = {("Alpha", "t1", 0): {"score": 3}, ("Alpha", "t1", 1): {"score": 9}}
        self.judge_config = JudgeConfig()
        self.judge_config.evaluation_delay_seconds = 0.0
        self.results = [TestCaseResult(test_id="t1", prompt="p", criteria="c")]

    def test_narrowing_to_best_keeps_the_wider_analyses_already_stored(self):
        stored = {
            ("Alpha", "t1", 0): {"type": "reasoning"},
            ("Alpha", "t1", 1): {"type": "reasoning"},
        }
        self.judge_config.reasoning_analysis = REASONING_SCOPE_BEST
        with patch("prompttestenv.analysis.analyze_reasoning") as mock_analyze:
            analyzed = run_analysis_phase(
                self.results, self.judge_config, "proj", _state(self.gen, self.evals, stored)
            )
        mock_analyze.assert_not_called()
        self.assertEqual(set(analyzed), set(stored))

    def test_widening_to_all_analyzes_only_the_missing_repetition(self):
        stored = {("Alpha", "t1", 1): {"type": "reasoning"}}
        self.judge_config.reasoning_analysis = REASONING_SCOPE_ALL
        with patch("prompttestenv.analysis.analyze_reasoning") as mock_analyze, \
                patch("prompttestenv.analysis.append_event"):
            mock_analyze.return_value = ReasoningStats()
            run_analysis_phase(
                self.results, self.judge_config, "proj", _state(self.gen, self.evals, stored)
            )
        self.assertEqual(mock_analyze.call_count, 1)


class TestAttachReasoningAgreesWithTheScope(unittest.TestCase):
    """The trace on screen must be the one the analysis phase chose to measure."""

    def test_best_repetition_is_the_same_one_both_places(self):
        gen = {
            ("Alpha", "t1", 0): _gen(reasoning_text="losing trace"),
            ("Alpha", "t1", 1): _gen(reasoning_text="winning trace"),
        }
        evals = {("Alpha", "t1", 0): {"score": 2}, ("Alpha", "t1", 1): {"score": 9}}
        chosen = keys_to_analyze(_state(gen, evals), REASONING_SCOPE_BEST, False)
        self.assertEqual(chosen, [("Alpha", "t1", 1)])

        row = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        row.candidates_perf["Alpha"] = CandidatePerformance()
        reasoning = {key: {"type": "reasoning"} for key in chosen}
        # Both runner.py call sites run attach_best right before
        # attach_reasoning; exercising the real sequence here, not just
        # attach_reasoning alone, is what would have caught the two
        # disagreeing before this test was written to assume they wouldn't.
        attach_best([row], evals, gen)
        attach_reasoning([row], reasoning, evals, gen)

        perf = row.candidates_perf["Alpha"]
        self.assertEqual(perf.best_output, "answer")
        self.assertEqual(perf.best_reasoning_text, "winning trace")
        self.assertIsNotNone(perf.best_reasoning_analysis)

    def test_no_trace_shown_when_the_actual_winner_has_none(self):
        """The bug this fixes: a winner with no trace used to borrow one.

        Rep 1 wins on score but has no reasoning trace at all. Before the
        fix, attach_reasoning restricted its own _best_key search to traced
        keys and picked rep 0's trace instead — showing it next to rep 1's
        response, a trace that belongs to a different, losing repetition.
        """
        gen = {
            ("Alpha", "t1", 0): _gen(output="loser", reasoning_text="a trace that exists"),
            ("Alpha", "t1", 1): _gen(output="winner", reasoning_text=""),
        }
        evals = {("Alpha", "t1", 0): {"score": 2}, ("Alpha", "t1", 1): {"score": 9}}
        row = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        row.candidates_perf["Alpha"] = CandidatePerformance()
        reasoning = {("Alpha", "t1", 0): {"type": "reasoning"}}

        attach_best([row], evals, gen)
        attach_reasoning([row], reasoning, evals, gen)

        perf = row.candidates_perf["Alpha"]
        self.assertEqual(perf.best_output, "winner")
        self.assertEqual(perf.best_reasoning_text, "")
        self.assertIsNone(perf.best_reasoning_analysis)


class TestAttachBest(unittest.TestCase):
    """The single computation of "which repetition wins", response side."""

    def _row(self, test_id="t1"):
        row = TestCaseResult(test_id=test_id, prompt="p", criteria="c")
        row.candidates_perf["Alpha"] = CandidatePerformance()
        return row

    def test_picks_the_highest_scoring_repetition(self):
        gen = {
            ("Alpha", "t1", 0): _gen(output="mediocre"),
            ("Alpha", "t1", 1): _gen(output="great"),
        }
        evals = {
            ("Alpha", "t1", 0): {"score": 4, "reason": "meh", "g_reason": "meh"},
            ("Alpha", "t1", 1): {"score": 9, "reason": "excellent", "g_reason": "excellent"},
        }
        row = self._row()
        attach_best([row], evals, gen)
        perf = row.candidates_perf["Alpha"]
        self.assertEqual(perf.best_output, "great")
        self.assertEqual(perf.best_reason, "excellent")

    def test_a_tie_goes_to_the_earliest_repetition(self):
        gen = {
            ("Alpha", "t1", 0): _gen(output="first"),
            ("Alpha", "t1", 1): _gen(output="second"),
        }
        evals = {k: {"score": 7, "reason": "ok", "g_reason": "ok"} for k in gen}
        row = self._row()
        attach_best([row], evals, gen)
        self.assertEqual(row.candidates_perf["Alpha"].best_output, "first")

    def test_winner_without_reasoning_trace_still_wins_the_response(self):
        """The disagreement attach_reasoning used to hide: see that test too.

        attach_best does not look at reasoning at all, so a repetition with
        no trace is exactly as eligible to win as one with a trace — the
        restriction belongs to attach_reasoning, applied on top afterwards.
        """
        gen = {
            ("Alpha", "t1", 0): _gen(output="untraced winner", reasoning_text=""),
            ("Alpha", "t1", 1): _gen(output="traced but worse"),
        }
        evals = {
            ("Alpha", "t1", 0): {"score": 9, "reason": "excellent", "g_reason": "excellent"},
            ("Alpha", "t1", 1): {"score": 3, "reason": "weak", "g_reason": "weak"},
        }
        row = self._row()
        attach_best([row], evals, gen)
        self.assertEqual(row.candidates_perf["Alpha"].best_output, "untraced winner")

    def test_all_evaluations_failed_still_shows_the_earliest_repetitions_text(self):
        """The one deliberate behavior change: no more silent blank best_output.

        Every score is -1 ("not measured"), so nothing is a real evaluated
        winner — but the response text is real, and the -1 rendered beside it
        already says the evaluation failed. Showing it removes a special case
        instead of adding one, and beats leaving the failure invisible.
        """
        gen = {
            ("Alpha", "t1", 0): _gen(output="first attempt"),
            ("Alpha", "t1", 1): _gen(output="second attempt"),
        }
        evals = {k: {"score": -1, "reason": "Error: judge down", "g_reason": "N/A"} for k in gen}
        row = self._row()
        attach_best([row], evals, gen)
        perf = row.candidates_perf["Alpha"]
        self.assertEqual(perf.best_output, "first attempt")
        self.assertEqual(perf.best_reason, "Error: judge down")

    def test_a_candidate_with_no_generated_keys_is_left_untouched(self):
        row = self._row()
        attach_best([row], {}, {})
        perf = row.candidates_perf["Alpha"]
        self.assertEqual(perf.best_output, "")

    def test_only_considers_keys_for_this_candidate_and_test(self):
        gen = {
            ("Alpha", "t1", 0): _gen(output="alpha t1"),
            ("Beta", "t1", 0): _gen(output="beta t1"),
            ("Alpha", "t2", 0): _gen(output="alpha t2"),
        }
        evals = {k: {"score": 5, "reason": "ok", "g_reason": "ok"} for k in gen}
        row = self._row("t1")
        row.candidates_perf["Beta"] = CandidatePerformance()
        attach_best([row], evals, gen)
        self.assertEqual(row.candidates_perf["Alpha"].best_output, "alpha t1")
        self.assertEqual(row.candidates_perf["Beta"].best_output, "beta t1")


class TestReasoningEnabled(unittest.TestCase):
    """"none" is a truthy string: every gate has to go through the property."""

    def test_none_scope_is_disabled_despite_being_truthy(self):
        config = JudgeConfig(reasoning_analysis=REASONING_SCOPE_NONE)
        self.assertTrue(config.reasoning_analysis, "the string itself is truthy")
        self.assertFalse(config.reasoning_enabled)

    def test_best_and_all_scopes_are_enabled(self):
        for scope in (REASONING_SCOPE_BEST, REASONING_SCOPE_ALL):
            with self.subTest(scope=scope):
                self.assertTrue(JudgeConfig(reasoning_analysis=scope).reasoning_enabled)


if __name__ == "__main__":
    unittest.main()

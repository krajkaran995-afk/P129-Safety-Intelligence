
import unittest

from src.recommendations import get_safety_recommendations


class SafetyRecommendationsTests(unittest.TestCase):

    def test_ergonomic_narrative_returns_relevant_recommendations(self):
        results = get_safety_recommendations(
            root_cause="Design Deficiency",
            incident_type="Ergonomic Injury",
            risk_level="Moderate",
            narrative=(
                "Operator experienced strain while using a pneumatic driver "
                "in a fixed position."
            ),
        )

        combined = " ".join(results).lower()

        self.assertTrue(results)
        self.assertTrue(
            any(
                word in combined
                for word in ["ergonomic", "posture", "fixed position"]
            )
        )

    def test_design_deficiency_triggers_engineering_review(self):
        results = get_safety_recommendations(
            root_cause="Design Deficiency",
            incident_type="Equipment Failure",
            risk_level="High",
            narrative=(
                "Equipment has no built-in stop from the working position."
            ),
        )

        combined = " ".join(results).lower()

        self.assertIn("engineering review", combined)
        self.assertIn("recorded risk classification", combined)

    def test_inadequate_training_returns_training_checks(self):
        results = get_safety_recommendations(
            root_cause="Inadequate Training",
            incident_type="Vehicle/Forklift Incident",
            risk_level="Medium",
            narrative="The worker had not completed the required induction.",
        )

        combined = " ".join(results).lower()

        self.assertTrue(
            any(
                word in combined
                for word in ["induction", "training records", "competency"]
            )
        )

    def test_empty_inputs_return_a_fallback(self):
        results = get_safety_recommendations(
            root_cause=None,
            incident_type=None,
            risk_level=None,
            narrative=None,
        )

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)

    def test_recommendations_do_not_contain_duplicates(self):
        results = get_safety_recommendations(
            root_cause="Poor Maintenance",
            incident_type="Equipment Failure",
            risk_level="Critical",
            narrative="Equipment failure occurred after maintenance was missed.",
        )

        self.assertEqual(len(results), len(set(results)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

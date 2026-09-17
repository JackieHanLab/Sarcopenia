import unittest

from sklearn.linear_model import LogisticRegression

from sarcopenia_face.development import (
    branch_sex_label_strata,
    cross_validated_candidate_summary,
    fit_training_rfe,
    make_locked_split,
    make_training_rfe_pipeline,
)
from sarcopenia_face.evaluation import evaluate_paired_predictions
from sarcopenia_face.scoring import bmi_branch, equal_weight_fusion
from sarcopenia_face.screening import threshold_screening_metrics


class ScoringTests(unittest.TestCase):
    def test_bmi_cutoff_routes_exact_boundary_to_higher_branch(self):
        self.assertEqual(bmi_branch(23.999), "lower")
        self.assertEqual(bmi_branch(24.0), "higher")

    def test_equal_weight_fusion(self):
        self.assertAlmostEqual(equal_weight_fusion(0.2, 0.8), 0.5)

    def test_invalid_bmi_and_scores_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bmi_branch(value)
        for value in (-0.01, 1.01, float("nan"), float("inf"), "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                equal_weight_fusion(value, 0.5)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"bmi": "22.0", "label": "0", "score_3d": "0.1", "score_thermal": "0.2"},
            {"bmi": "23.0", "label": "1", "score_3d": "0.8", "score_thermal": "0.7"},
            {"bmi": "23.5", "label": "0", "score_3d": "0.2", "score_thermal": "0.3"},
            {"bmi": "23.8", "label": "1", "score_3d": "0.9", "score_thermal": "0.8"},
            {"bmi": "24.0", "label": "0", "score_3d": "0.1", "score_thermal": "0.2"},
            {"bmi": "25.0", "label": "1", "score_3d": "0.8", "score_thermal": "0.7"},
            {"bmi": "27.0", "label": "0", "score_3d": "0.2", "score_thermal": "0.3"},
            {"bmi": "30.0", "label": "1", "score_3d": "0.9", "score_thermal": "0.8"},
        ]

    def test_reports_primary_auc_by_branch_on_the_paired_sample(self):
        results = evaluate_paired_predictions(self.rows)
        self.assertEqual([row["bmi_branch"] for row in results], ["lower", "higher"])
        self.assertEqual([row["n"] for row in results], [4, 4])
        for row in results:
            self.assertEqual(row["auc_3d"], 1.0)
            self.assertEqual(row["auc_thermal"], 1.0)
            self.assertEqual(row["auc_fusion"], 1.0)
            self.assertEqual(row["aupr_3d"], 1.0)
            self.assertEqual(row["aupr_thermal"], 1.0)
            self.assertEqual(row["aupr_fusion"], 1.0)
            for metric in ("auc_3d", "auc_thermal", "auc_fusion"):
                self.assertLessEqual(row[f"{metric}_ci95_low"], row[metric])
                self.assertGreaterEqual(row[f"{metric}_ci95_high"], row[metric])
                self.assertGreaterEqual(row[f"{metric}_ci95_low"], 0.0)
                self.assertLessEqual(row[f"{metric}_ci95_high"], 1.0)

    def test_delong_auc_matches_rank_auc_with_tied_scores(self):
        rows = [
            {"bmi": "22.0", "label": "0", "score_3d": "0.1", "score_thermal": "0.2"},
            {"bmi": "23.0", "label": "0", "score_3d": "0.5", "score_thermal": "0.4"},
            {"bmi": "23.5", "label": "1", "score_3d": "0.5", "score_thermal": "0.6"},
            {"bmi": "23.8", "label": "1", "score_3d": "0.9", "score_thermal": "0.8"},
        ]
        result = evaluate_paired_predictions(rows)[0]
        self.assertEqual(result["auc_3d"], 0.875)
        expected_low = 0.875 - 1.959963984540054 * (0.03125**0.5)
        self.assertAlmostEqual(result["auc_3d_ci95_low"], expected_low)
        self.assertEqual(result["auc_3d_ci95_high"], 1.0)

    def test_missing_paired_score_is_not_silently_dropped(self):
        rows = [dict(self.rows[0], score_thermal="")]
        with self.assertRaises(ValueError):
            evaluate_paired_predictions(rows)

    def test_branch_without_both_classes_is_rejected(self):
        rows = [dict(self.rows[0], bmi="23.0"), dict(self.rows[2], bmi="23.5")]
        with self.assertRaises(ValueError):
            evaluate_paired_predictions(rows)


class DevelopmentAndScreeningTests(unittest.TestCase):
    def test_locked_split_uses_branch_sex_label_strata(self):
        bmi = [22.0, 22.1, 22.2, 22.3, 25.0, 25.1, 25.2, 25.3] * 2
        labels = [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
        sex = ["female"] * 8 + ["male"] * 8
        strata = branch_sex_label_strata(bmi, labels, sex)
        train_index, test_index = make_locked_split(
            bmi, labels, sex, test_size=0.5, random_state=7
        )
        self.assertEqual(len(strata), 16)
        self.assertFalse(set(train_index) & set(test_index))
        self.assertEqual(len(train_index) + len(test_index), 16)

    def test_rfe_is_fit_to_training_data(self):
        features = [
            [0.0, 0.1, 1.0],
            [0.1, 0.2, 0.0],
            [0.9, 0.8, 1.0],
            [1.0, 0.9, 0.0],
        ]
        pipeline = fit_training_rfe(
            features,
            [0, 0, 1, 1],
            LogisticRegression(solver="liblinear"),
            n_features_to_select=2,
        )
        self.assertEqual(pipeline.transform(features).shape, (4, 2))

    def test_cross_validation_reports_auc_and_aupr_without_model_selection(self):
        features = [
            [0.0, 0.1, 1.0],
            [0.1, 0.2, 0.0],
            [0.2, 0.3, 1.0],
            [0.3, 0.4, 0.0],
            [0.7, 0.6, 1.0],
            [0.8, 0.7, 0.0],
            [0.9, 0.8, 1.0],
            [1.0, 0.9, 0.0],
        ]
        candidate = make_training_rfe_pipeline(
            LogisticRegression(solver="liblinear"), n_features_to_select=2
        )
        results = cross_validated_candidate_summary(
            features,
            [0, 0, 0, 0, 1, 1, 1, 1],
            {"linear_logistic": candidate},
            folds=2,
            random_state=7,
        )
        self.assertEqual(results[0]["candidate"], "linear_logistic")
        self.assertIn("cv_roc_auc_mean", results[0])
        self.assertIn("cv_aupr_mean", results[0])

    def test_screening_metrics_follow_the_referral_rule(self):
        results = threshold_screening_metrics(
            [0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], [0.5]
        )
        self.assertEqual(results[0]["sensitivity"], 1.0)
        self.assertEqual(results[0]["specificity"], 0.5)
        self.assertEqual(results[0]["referral_proportion"], 0.75)


if __name__ == "__main__":
    unittest.main()

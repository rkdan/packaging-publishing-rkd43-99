import unittest
from cancer_prediction_rkd43_99.cancer_model import CancerModel


class TestCancerModel(unittest.TestCase):

    def test_diagnosis_to_target(self):
        model = CancerModel()
        diagnosis = "Malignant"
        target = model.diagnosis_to_target(diagnosis)
        self.assertEqual(0, target)

        diagnosis = " "
        target = model.diagnosis_to_target(diagnosis)
        self.assertEqual(1, target)

    def test_target_to_diagnosis(self):
        model = CancerModel()
        target = 0
        diagnosis = model.target_to_diagnosis(target)
        self.assertEqual("Malignant", diagnosis)
  
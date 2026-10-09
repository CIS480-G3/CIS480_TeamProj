# import script fix (TODO address this repo-wide so I don't need this anymore)
import sys
from pathlib import Path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_dir))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sklearn.metrics import confusion_matrix

# Import functions from confusion_matrix_numpy.py
from confusion_matrix_numpy import numpy_confusion_matrix, run_evaluation

def test_np_matrix_vs_sklearn():
    # call eval and save results
    pred_default, pred_calc, y_test = run_evaluation()
    
    # calc numpy confusion matrix
    np_cm_default = numpy_confusion_matrix(y_test, pred_default)
    np_cm_calc = numpy_confusion_matrix(y_test, pred_calc)
    
    # calc scikit-learn confusion matrix
    sklearn_cm_default = confusion_matrix(y_test, pred_default)
    sklearn_cm_calc = confusion_matrix(y_test, pred_calc)
    
    # assert that outputs are structurally and numerically identical
    np.testing.assert_array_equal(np_cm_default, sklearn_cm_default)
    np.testing.assert_array_equal(np_cm_calc, sklearn_cm_calc)
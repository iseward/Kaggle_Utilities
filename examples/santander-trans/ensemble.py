import config
import jhkaggle
from jhkaggle.ensemble_glm import ensemble
import time
import sklearn

def run_ensemble():
  MODELS = [
  #  'xgboost-0p852964_20210217-233940',
    'keras-0p847074_20210218-012754',
    'rforest-0p77575_20210218-093834',
    'extree-0p852162_20210218-094539',
    'adaboost-0p884786_20210218-160138',
    'knn-0p538086_20210218-201302'
  ]
  ensemble(MODELS)


if __name__ == "__main__":
    run_ensemble()


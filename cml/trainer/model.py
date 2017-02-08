import json
import os

import google.cloud.ml.features as features


def runs_on_cloud():
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  return env.get('task', None)

class cmlFeatures(object):

  csv_columns = ('time', 'inserted_at', 'value')

  time = features.key('time')
  value = features.target('value').discrete()
  measurements = [
      features.numeric('inserted_at'), features.numeric('value')
  ]


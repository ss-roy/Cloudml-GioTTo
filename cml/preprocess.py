"""cml Classification Sample Preprocessing Runner.
"""

import argparse
import datetime
import os
import subprocess
import sys

import apache_beam as beam
import trainer.model as cml
#from trainer import model as cml

import google.cloud.ml as ml
import google.cloud.ml.io as io

# Model variables
MODEL_NAME = 'cml'
TRAINER_NAME = 'trainer-1.0.tar.gz'

def _default_project():
  get_project = ['gcloud', 'config', 'list', 'project',
                 '--format=value(core.project)']

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()

def parse_arguments(argv):
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on the cml model data.')
  parser.add_argument('--project_id',
                      help='The project to which the job will be submitted.')
  parser.add_argument('--cloud', action='store_true',
                      help='Run preprocessing on the cloud.')
  parser.add_argument('--training_data',
                      default='gs://cloud-ml-data/cml/data_train.csv',
                      help='Data to analyze and encode as training features.')
  parser.add_argument('--eval_data',
                      default='gs://cloud-ml-data/cml/data_eval.csv',
                      help='Data to encode as evaluation features.')
  parser.add_argument('--predict_data',
                      default='gs://cloud-ml-data/cml/data_predict.csv',
                      help='Data to encode as prediction features.')
  parser.add_argument('--output_dir', default=None,
                      help=('Google Cloud Storage or Local directory in which '
                            'to place outputs.'))
  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud and not args.project_id:
    args.project_id = _default_project()

  if not args.output_dir:
    if args.cloud:
      args.output_dir = os.path.join('gs://' + args.project_id + '-ml',
                                     MODEL_NAME)
    else:
      path = 'output'
      if not os.path.isdir(path):
        os.makedirs(path)
      args.output_dir = path

  args.trainer_uri = os.path.join(args.output_dir, TRAINER_NAME)
  return args

def preprocess(pipeline, training_data, eval_data, predict_data, output_dir):
  feature_set = cml.cmlFeatures()

  training_data_source = beam.io.TextFileSource(
      training_data, strip_trailing_newlines=True,
      coder=io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns))

  eval_data_source = beam.io.TextFileSource(
      eval_data, strip_trailing_newlines=True,
      coder=io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns))

  predict_data_source = beam.io.TextFileSource(
      predict_data, strip_trailing_newlines=True,
      coder=io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns,
                                         has_target_columns=False))
  train = pipeline | beam.Read('ReadTrainingData', training_data_source)
  evaluate = pipeline | beam.Read('ReadEvalData', eval_data_source)
  predict = pipeline | beam.Read('ReadPredictData', predict_data_source)

  # TODO(b/32726166) Update input_format and format_metadata to read from these
  # values directly from the coder.
  (metadata, train_features, eval_features, predict_features) = (
      (train, evaluate, predict)
      | 'Preprocess'
      >> ml.Preprocess(feature_set, input_format='csv',
                       format_metadata={'headers': feature_set.csv_columns}))

  # Writes metadata.yaml, features_train, features_eval, and features_eval files
  # pylint: disable=expression-not-assigned
  (metadata | 'SaveMetadata' >> io.SaveMetadata(os.path.join(output_dir,
                                                             'metadata.yaml')))

  # We turn off sharding of these feature files because the dataset very small.
  (train_features | 'SaveTrain'
                  >> io.SaveFeatures(
                      os.path.join(output_dir, 'features_train')))
  (eval_features | 'SaveEval'
                 >> io.SaveFeatures(
                     os.path.join(output_dir, 'features_eval')))
  (predict_features | 'SavePredict'
                    >> io.SaveFeatures(
                        os.path.join(output_dir, 'features_predict')))
  # pylint: enable=expression-not-assigned

  return metadata, train_features, eval_features, predict_features


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)

  if args.cloud:
    print 'Building',TRAINER_NAME,'package.'
    subprocess.check_call(['python', 'setup.py', 'sdist', '--format=gztar'])
    subprocess.check_call(['gsutil', '-q', 'cp',
                           os.path.join('dist', TRAINER_NAME),
                           args.trainer_uri])
    options = {
        'staging_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'job_name': ('cloud-ml-sample-cml-preprocess' + '-'
                     + datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        'project': args.project_id,
        'extra_packages': [ml.sdk_location, args.trainer_uri],
        'teardown_policy': 'TEARDOWN_ALWAYS',
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    p = beam.Pipeline('BlockingDataflowPipelineRunner', options=opts)
  else:
    p = beam.Pipeline('DirectPipelineRunner')

  preprocess(
      pipeline=p,
      training_data=args.training_data,
      eval_data=args.eval_data,
      predict_data=args.predict_data,
      output_dir=args.output_dir)

  p.run()


if __name__ == '__main__':
  main()













	

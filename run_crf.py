# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import collections
import os
import time
import math
import json
import six
import random
import gc

import numpy as np

import crf_util

import pickle

import tensorflow as tf
import sentencepiece as spm
import function_builder_GPU as function_builder
import model_utils as model_utils
import acc_f1

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.05,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.05,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length.")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a vector.")
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("cache_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("predict_dir", default="",
                    help="Dir for predictions.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")

# Data preprocessing config
flags.DEFINE_integer("max_seq_length",
                     default=512, help="Max sequence length")
flags.DEFINE_bool("lower", default=True, help="Use uncased data.")

# Training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_batch_size", default=48,
                     help="batch size for training")
flags.DEFINE_integer("train_steps", default=0,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. "
                          "Use 0 to save all.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")

# Optimization
flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
flags.DEFINE_float("lr_layer_decay_rate", default=0.9,
                   help="Top layer: lr[L] = FLAGS.learning_rate."
                        "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")

# Eval / Prediction
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do predict")
flags.DEFINE_integer("eval_batch_size", default=32,
                     help="batch size for eval")
flags.DEFINE_integer("eval_steps", default=100,
                     help="do eval steps")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# crf & label
flags.DEFINE_integer("crf_classes", default=0, help="")
flags.DEFINE_string("label_mode", default="normal", help="")
flags.DEFINE_string("label_mask", default="normal", help="")
flags.DEFINE_bool("no_crf", default=True, help="")
flags.DEFINE_string("task", default="conll2003",
                    help="")

FLAGS = flags.FLAGS


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_x_id": tf.FixedLenFeature([seq_length], tf.int64),
        "label_gather": tf.FixedLenFeature([seq_length], tf.int64),
        "label_mask_x": tf.FixedLenFeature([seq_length], tf.float32),
        "label_mask_gather": tf.FixedLenFeature([seq_length], tf.float32),
        "label_index": tf.FixedLenFeature([seq_length], tf.int64)
    }

    tf.logging.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params, input_context=None):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        elif FLAGS.do_eval:
            batch_size = FLAGS.eval_batch_size
        else:
            batch_size = 1

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if input_context is not None:
            tf.logging.info("Input pipeline id %d out of %d",
                            input_context.input_pipeline_id, input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def get_model_fn():
    def model_fn(features, labels, mode, params):
        # Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, logits, label, mask = function_builder.get_crf_outputs(
            FLAGS, features, is_training)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # predict mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"logits": logits,
                           "labels": label,
                           'mask': features['input_mask'],
                           'label_mask': mask}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
            return output_spec



        # Evaluation mode
        elif mode == tf.estimator.ModeKeys.EVAL:
            assert FLAGS.num_hosts == 1

            def metric_fn(per_example_loss, label_ids, logits, weight):
                eval_input_dict = {
                    'labels': label_ids,
                    'predictions': logits,
                    'weights': weight
                }

                accuracy = tf.metrics.accuracy(**eval_input_dict)

                eval_input_dict = {
                    'labels': tf.one_hot(label_ids, FLAGS.crf_classes),
                    'predictions': tf.one_hot(logits, FLAGS.crf_classes),
                    'weights': weight
                }
                f1 = tf.contrib.metrics.f1_score(**eval_input_dict)
                loss = tf.metrics.mean(values=per_example_loss)
                return {'eval_accuracy': accuracy,
                        'eval_loss': loss,
                        'f1': f1}

            metric_args = [per_example_loss, label, logits, mask]

            eval_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=metric_fn(*metric_args))

            return eval_spec

        # load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        train_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "conll2003": crf_util.Conll2003Processor,
        "bc5cdr": crf_util.BC5CDRProcessor,
        "anatem": crf_util.AnatEMProcessor,
        "bc2gm": crf_util.BC2GMProcessor,
        "bc4chemd": crf_util.BC4CHEMDProcessor,
        "bionlp09": crf_util.BioNLP09Processor,
        "bionlp11id": crf_util.BioNLP11IDProcessor,
        "bionlp11ep": crf_util.BioNLP11EPIProcessor,
        "bionlp13cg": crf_util.BioNLP13CGProcessor,
        "bionlp13ge": crf_util.BioNLP13GEProcessor,
        "bionlp13pc": crf_util.BioNLP13PCProcessor,
        "craft": crf_util.CRAFTProcessor,
        "exptm": crf_util.ExPTMProcessor,
        "jnlpba": crf_util.JNLPBAProcessor,
        "linnaeus": crf_util.linnaeusProcessor,
        "ncbi": crf_util.NCBIProcessor
    }
    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    processor = processors[FLAGS.task](FLAGS.spiece_model_file, FLAGS.lower, "data", FLAGS.cache_dir,
                                       FLAGS.max_seq_length)
    FLAGS.crf_classes = processor.classes
    if not FLAGS.train_steps:
        FLAGS.train_steps = processor.train_step
    FLAGS.model_dir = "{}_{}_{}_{}_{}".format(FLAGS.model_dir,
                                              FLAGS.task,
                                              "nocrf" if FLAGS.no_crf else "crf",
                                              FLAGS.label_mode,
                                              FLAGS.label_mask
                                              )

    ### GPU Configuration
    run_config = model_utils.configure_tpu(FLAGS)
    model_fn = get_model_fn()

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_file = processor.get_train_data()
        if not tf.gfile.Exists(train_file):
            raise ValueError("no train file")
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    steps_and_files = []
    filenames = tf.gfile.ListDirectory(FLAGS.model_dir)
    for filename in filenames:
        if filename.endswith(".index"):
            ckpt_name = filename[:-6]
            cur_filename = os.path.join(FLAGS.model_dir, ckpt_name)
            global_step = int(cur_filename.split("-")[-1])
            tf.logging.info("Add {} to eval list.".format(cur_filename))
            steps_and_files.append([global_step, cur_filename])
    steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

    if FLAGS.do_eval:
        eval_file = processor.get_dev_data()
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        eval_results = []
        for global_step, filename in steps_and_files:
            ret = estimator.evaluate(
                input_fn=eval_input_fn,
                checkpoint_path=filename)

            ret["step"] = global_step
            ret["path"] = filename

            eval_results.append(ret)

            tf.logging.info("=" * 80)
            log_str = "Eval result | "
            for key, val in sorted(ret.items(), key=lambda x: x[0]):
                log_str += "{} {} | ".format(key, val)
            tf.logging.info(log_str)

        with open(os.path.join(FLAGS.predict_dir, "{}_eval.txt".format(FLAGS.model_dir)), "w") as f:
            for ret in eval_results:
                log_str = "Eval result : "
                for key, val in sorted(ret.items(), key=lambda x: x[0]):
                    log_str += "{} {} \n ".format(key, val)
            f.write(log_str)

    if FLAGS.do_predict:
        f = open(os.path.join(FLAGS.predict_dir, "{}.txt".format(FLAGS.model_dir)), "w")
        pred_file = processor.get_test_data()

        pred_input_fn = file_based_input_fn_builder(
            input_file=pred_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        for global_step, filename in steps_and_files:
            predict_results = []
            for pred_cnt, result in enumerate(estimator.predict(
                    input_fn=pred_input_fn,
                    yield_single_examples=True,
                    checkpoint_path=filename)):
                if pred_cnt % 100 == 0:
                    tf.logging.info("Predicting submission for example: {}".format(
                        pred_cnt))
                for key in result.keys():
                    result[key] = result[key].tolist()
                predict_results.append(result)

            predict_json_path = os.path.join(FLAGS.predict_dir, "{}_{}.json".format(
                FLAGS.model_dir, global_step))

            with tf.gfile.Open(predict_json_path, "w") as fp:
                json.dump(predict_results, fp, indent=4)

            f.write("%d\n" % global_step)
            acc_f1.get_result(predict_json_path, f, processor.decode_map)
        f.close()

    def _remove_checkpoint(checkpoint_path):
        for ext in ["meta", "data-00000-of-00001", "index"]:
            src_ckpt = checkpoint_path + ".{}".format(ext)
            tf.logging.info("removing {}".format(src_ckpt))
            tf.gfile.Remove(src_ckpt)

    for global_step, filename in steps_and_files[:-1]:
        _remove_checkpoint(filename)


if __name__ == "__main__":
    tf.app.run()

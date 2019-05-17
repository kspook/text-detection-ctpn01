from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
from tensorflow.python.platform import gfile
import glob
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

dir_path = os.path.dirname(os.path.realpath(__file__))


def export():
  cfg_from_file(os.path.join(dir_path, 'text.yml'))
  config = tf.ConfigProto(allow_soft_placement=True)
  sess = tf.Session(config=config)
  # with gfile.FastGFile('../data/ctpn.pb', 'rb') as f:
  #   graph_def = tf.GraphDef()
  #   graph_def.ParseFromString(f.read())
  #   sess.graph.as_default()
  #   tf.import_graph_def(graph_def, name='')
  # sess.run(tf.global_variables_initializer())

  net = get_network("VGGnet_test")
  print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
  saver = tf.train.Saver()
  print(saver)  
  try:
    ckpt_path = os.path.abspath(os.path.join(dir_path, cfg.TEST.checkpoints_path))
    print('check_path, ', ckpt_path)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('done')
  except:
    raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
  print(' done.')

  input_img = sess.graph.get_tensor_by_name('Placeholder:0')
  output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
  output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

  builder = tf.saved_model.builder.SavedModelBuilder('./export/1')

  imageplaceholder_info = tf.saved_model.utils.build_tensor_info(input_img)
  cls_prob_info = tf.saved_model.utils.build_tensor_info(output_cls_prob)
  box_pred_info = tf.saved_model.utils.build_tensor_info(output_box_pred)
  print('predict_method_name,', tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
        'image': imageplaceholder_info
      },
      outputs={
        'output_cls_prob': cls_prob_info,
        'output_box_pred': box_pred_info
      },
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
  )
  init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
       signature_def_map={'ctpn_recs_predict': prediction_signature}, legacy_init_op=init_op)
  builder.save()


if __name__ == '__main__':
  export()

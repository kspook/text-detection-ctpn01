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




def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  #  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  image = tf.image.convert_image_dtype(image_buffer, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  image = tf.image.central_crop(image, central_fraction=0.875)
  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  '''
  image = tf.image.resize_bilinear(
      image, [150, 100], align_corners=False)
      #image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
  '''
  print('tf.resize')
  image = tf.squeeze(image, [0])
  print('tf.squeeze')
  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

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
  '''
  input_img = sess.graph.get_tensor_by_name('Placeholder:0')
  output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
  output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
  '''

  raw_image =  tf.placeholder(tf.float32, shape=[None, None, None, 3])  #\u8f93\u5165\u539f\u59cb\u56fe\u50cf
  print('raw_image, ')
  jpeg = preprocess_image(raw_image)  #\u9884\u5904\u7406,\u7f29\u653e
  #jpeg,im_info = preprocess_image(raw_image)  #\u9884\u5904\u7406,\u7f29\u653e
  # Preprocessing our input image


  #cls_prob_info,box_pred_info = tf.import_graph_def\
  output_tensor_cls_prob,output_tensor_box_pred = tf.import_graph_def\
                          (tf.get_default_graph().as_graph_def(),
                           input_map={'Placeholder:0': jpeg},
                           return_elements=['Reshape_2:0','rpn_bbox_pred/Reshape_1:0'])

  builder = tf.saved_model.builder.SavedModelBuilder('./exportn/1')

  imageplaceholder_info = tf.saved_model.utils.build_tensor_info(jpeg)
  #imageplaceholder_info = tf.saved_model.utils.build_tensor_info(input_img)
  #cls_prob_info = tf.saved_model.utils.build_tensor_info(output_cls_prob)
  #box_pred_info = tf.saved_model.utils.build_tensor_info(output_box_pred)
  print('predict_method_name,', tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
        'image': imageplaceholder_info
      },
      outputs={
        'output_cls_prob': output_tensor_cls_prob,
        'output_box_pred': output_tensor_box_pred
        #'output_cls_prob': cls_prob_info,
        #'output_box_pred': box_pred_info
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

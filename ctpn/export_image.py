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
from lib.fast_rcnn.test import _get_blobs
from lib.rpn_msr.proposal_layer_tf import proposal_layer

dir_path = os.path.dirname(os.path.realpath(__file__))


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def preprocess_image(image_buffer):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""

    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.

    image = tf.image.decode_image(image_buffer, channels=3)
    image.set_shape([256, 256, 256,3])

    # self.img_pl = tf.placeholder(tf.string, name='input_image_as_bytes')
    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    image = tf.expand_dims(image, 0)
    image = tf.squeeze(image, [0])
    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def query_ctpn(sess,  cv2img):
    """Args:
        sess: tensorflow session
        cfg: CTPN config
        img: numpy array image

   Returns:
       A list of detected bounding boxes,
        each bounding box have followed coordinates: [(xmin, ymin), (xmax, ymax)]
            (xmin, ymin) -------------
                 |                    |
             ---------------- (xmax, ymax)
    """
    # Specify input/output
    input_img  = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_box = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    #print('query_pb : img, ',  img)

    img, scale = resize_im(cv2img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                                    dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_box, output_box_pred],
                                      feed_dict={input_img: blobs['data']})
        #print('cls_prob, ', cls_prob, box_pred )
        print('box_pred, ',  box_pred )
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'],
                                 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
        print('rois, ', rois)

        scores = rois[:, 0]
        #print('scores, ', scores )
        boxes = rois[:, 1:5] / im_scales[0]
        #print('boxes=rois, ', boxes )

        textdetector = TextDetector()
        print('textDetector, ', textdetector )
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        print('boxes=textdetector, ', boxes )

        # Convert boxes to bouding rectangles
        rects = []
        for box in boxes:
            min_x = min(int(box[0]/scale), int(box[2]/scale), int(box[4]/scale), int(box[6]/scale))
            min_y = min(int(box[1]/scale), int(box[3]/scale), int(box[5]/scale), int(box[7]/scale))
            max_x = max(int(box[0]/scale), int(box[2]/scale), int(box[4]/scale), int(box[6]/scale))
            max_y = max(int(box[1]/scale), int(box[3]/scale), int(box[5]/scale), int(box[7]/scale))

        rects.append([(min_x, min_y), (max_x, max_y)])
        print('rects.append, ', rects)
        return rects


def export():
  '''
  No 1 Sess outf of 2 : ctpn_sess
  '''
  cfg_from_file(os.path.join(dir_path, 'text_post.yml'))
  config = tf.ConfigProto(allow_soft_placement=True)
  ctpn_sess = tf.Session(config=config)
  with ctpn_sess.as_default():
   with tf.gfile.FastGFile('../data/ctpn.pb', 'rb') as f:
     graph_def = tf.GraphDef()
     graph_def.ParseFromString(f.read())
     ctpn_sess.graph.as_default()
     tf.import_graph_def(graph_def, name='')
   ctpn_sess.run(tf.global_variables_initializer())


  cv2img = cv2.imread("../data/demo/006.jpg", cv2.IMREAD_COLOR)

  result_boxes=query_ctpn(ctpn_sess,  cv2img) 

  print('Creating boxes done')
  '''
  No 2 Sess outf of 2:sess
  '''
  with tf.Session() as sess:
     with gfile.FastGFile('../data/ctpn.pb', 'rb') as f:
      restored_graph_def = tf.GraphDef()
      restored_graph_def.ParseFromString(f.read())
      tf.import_graph_def(
         restored_graph_def, 
         input_map=None,
         return_elements=None,
         name=""
      )

  '''
  export_path_base = args.export_model_dir
  export_path = os.path.join(tf.compat.as_bytes(export_path_base),
  tf.compat.as_bytes(str(args.model_version)))
  '''
  builder = tf.saved_model.builder.SavedModelBuilder('../exportPo/1')
  #print('Exporting trained model to', export_path)
  print('Exporting trained model ')

  raw_image =  tf.placeholder(tf.string,  name='tf_box')  
  feature_configs = {
        'image/encoded': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
  }
  tf_example = tf.parse_example(raw_image , feature_configs)
    
  jpegs = tf_example['image/encoded']
  image_string = tf.reshape(jpegs, shape=[])
  jpeg= preprocess_image(image_string)  
  print('jpeg,jpeg.shape[]', jpeg, jpeg.shape)

  output_tensor_cls_prob,output_tensor_box_pred = tf.import_graph_def\
                            (tf.get_default_graph().as_graph_def(),
                           input_map={'Placeholder:0': jpeg},
                           return_elements=['Reshape_2:0','rpn_bbox_pred/Reshape_1:0'])

  tensor_info_input = tf.saved_model.utils.build_tensor_info(raw_image)
  tensor_info_output_cls_prob = tf.saved_model.utils.build_tensor_info(output_tensor_cls_prob)
  tensor_info_output_box_pred = tf.saved_model.utils.build_tensor_info(output_tensor_box_pred)

  '''
  #crop_resize_img,crop_resize_im_info = resize_im(cv2img, result_boxes)
  #crop_resize_img,crop_resize_im_info = crop_resize_image(imageplaceholder_info, result_boxes)
  # output_crop_resize_img = tf.saved_model.utils.build_tensor_info(crop_resize_img)
  #output_crop_resize_img_info = tf.saved_model.utils.build_tensor_info(crop_resize_im_info)
  #----------
  '''
  result_boxes= np.array(result_boxes, dtype=np.float32)
  result_boxes= tf.convert_to_tensor(result_boxes)
  tensor_info_output_boxes = tf.saved_model.utils.build_tensor_info(result_boxes)

  prediction_post_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
           inputs={'images': tensor_info_input},
           outputs={'detection_boxes': tensor_info_output_boxes},
           #outputs={'detection_boxes': tensor_info_output_boxes,
          #      'resize_im_info':im_info_output,
          #      'crop_resize_img': output_crop_resize_img,
          #      'crop_resize_im_info': output_crop_resize_img_info,},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
  ))

  builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
        #  'predict_images':prediction_signature,
          'predict_images_post': prediction_post_signature
  })
  builder.save(as_text=False)

if __name__ == '__main__':
    export()


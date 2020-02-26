import tensorflow as tf

import functools

input_files = ["pet.tfrecord"]
filename_dataset = tf.data.Dataset.from_tensor_slices(input_files)
records_dataset = tf.data.TFRecordDataset(filename_dataset, buffer_size=8 * 1000 * 1000)

feature_dict = {
  'image/height': tf.io.FixedLenFeature([], tf.int64),
  'image/width': tf.io.FixedLenFeature([], tf.int64),
  'image/filename': tf.io.FixedLenFeature([], tf.string),
  'image/encoded': tf.io.FixedLenFeature([], tf.string),
  'image/format': tf.io.FixedLenFeature([], tf.string),
  'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
  'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
  'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
  'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
  'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
  'image/object/class/label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_dict)

parsed_image_dataset = records_dataset.map(_parse_image_function)
iter = parsed_image_dataset.make_one_shot_iterator()
i = 0
with tf.Session() as sess:
  try:
    while True:
      el = iter.get_next()

      heightT = el["image/height"]
      widthT = el["image/width"]
      encodedT = el["image/encoded"]
      xminT = el["image/object/bbox/xmin"]
      yminT = el["image/object/bbox/ymin"]
      xmaxT = el["image/object/bbox/xmax"]
      ymaxT = el["image/object/bbox/ymax"]

      height, width, encoded, xmin, ymin, xmax, ymax = \
        sess.run(
          [heightT, widthT, encodedT, xminT, yminT, xmaxT, ymaxT])

      fwrite = tf.io.write_file("original" + str(i) + ".jpg", encoded)
      sess.run(fwrite)

      decoded = tf.io.decode_jpeg(encoded)
      image = tf.image.convert_image_dtype(decoded, tf.float32)
      image = tf.image.draw_bounding_boxes([image], [[[ymin, xmin, ymax, xmax]]])
      image = tf.squeeze(image)
      image = tf.image.convert_image_dtype(image, tf.uint8)
      fwrite = tf.io.write_file("boxed" + str(i) + ".jpg", tf.io.encode_jpeg(image))
      sess.run(fwrite)
      i = i + 1
  except tf.errors.OutOfRangeError:
    pass

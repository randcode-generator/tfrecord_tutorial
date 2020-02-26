import tensorflow as tf
import os
import io
import PIL.Image
import re
from lxml import etree

def get_class_name_from_filename(file_name):
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def recursive_parse_xml_to_dict(xml):
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def dict_to_tf_example(data,
                       label_map_dict):
  img_path = os.path.join(os.getcwd(), "train_data", data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  if 'object' in data:
    for obj in data['object']:
      xmin = float(obj['bndbox']['xmin'])
      xmax = float(obj['bndbox']['xmax'])
      ymin = float(obj['bndbox']['ymin'])
      ymax = float(obj['bndbox']['ymax'])

      xmins.append(xmin / width)
      ymins.append(ymin / height)
      xmaxs.append(xmax / width)
      ymaxs.append(ymax / height)
      class_name = get_class_name_from_filename(data['filename'])
      classes_text.append(class_name.encode('utf8'))
      classes.append(label_map_dict[class_name])

  feature_dict = {
    'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
    'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
    'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
    'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
    'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
    'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
    'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
    'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

label_map_dict = {
    'Abyssinian' : 1,
    'american_bulldog': 2
}

tfrecordWriter = tf.python_io.TFRecordWriter("pet.tfrecord")
files = ['Abyssinian_10.xml', 'american_bulldog_10.xml']
for filename in files:
  xml_path = os.path.join(os.getcwd(), 'train_data', filename)

  with tf.gfile.GFile(xml_path, 'r') as fid:
    xml_str = fid.read()
  xml = etree.fromstring(xml_str)
  data = recursive_parse_xml_to_dict(xml)['annotation']
  tf_example = dict_to_tf_example(data, label_map_dict)

  tfrecordWriter.write(tf_example.SerializeToString())

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


output_directory = 'inference_graph'
labelmap_path = 'label_map.pbtxt'

import tensorflow as tf
from object_detection.utils import label_map_util
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'content/{output_directory}/saved_model')


# In[ ]:


import pandas as pd
import os
import glob
from pdf2image import convert_from_path

input_path = 'test/'
images = os.listdir(input_path)

for i in range(len(images)):
    if images[i][-4:] == '.jpg':
        pass
    elif images[i][-4:] == '.pdf':
        pages = convert_from_path(input_path+images[i])
        for page in pages:
            page.save(input_path +'pdf_image_' +str(i).zfill(3)+ '.jpg', 'JPEG')
            
images = sorted(glob.glob1(input_path, "*.jpg"))
images


# In[ ]:


from inferenceutils import *
image_np = image_np = load_image_into_numpy_array('test/' + 'Invoice_1.jpg')
output_dict = run_inference_for_single_image(model, image_np)
vis_util.visualize_boxes_and_labels_on_image_array(
image_np,
output_dict['detection_boxes'],
output_dict['detection_classes'],
output_dict['detection_scores'],
category_index,
instance_masks=output_dict.get('detection_masks_reframed', None),
use_normalized_coordinates=True,
line_thickness=3)
display(Image.fromarray(image_np))


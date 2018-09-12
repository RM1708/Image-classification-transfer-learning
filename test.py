
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
#import argparse
import os   #, sys
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


# In[2]:


t=tqdm(pd.read_csv('test.csv').values)
test=[]
i=0
for tt in t:
    test.append(tt[0])
    i+=1


# In[3]:


test


# In[4]:


def load_image(filename):
    #Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
    #Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_graph(filename):
    #Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


# In[5]:


def run_graph(src, dest, labels, \
              input_layer_name, \
              output_layer_name, \
              num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        # predictions  will contain a two-dimensional array, where one
        # dimension represents the input image count, and the other has
        # predictions per class
        i=0
        #with open('submit.csv','w') as outfile:
        outfile = open('submit.csv','w')
        for f in os.listdir(src):
            im=Image.open(os.path.join(src,f))
            img=im.convert('RGB')
            img.save(os.path.join(dest,test[i]+'.jpg'))
            print("File Index:", i)
            image_data=load_image(os.path.join(dest,test[i]+'.jpg'))
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
#            predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})
#            # Sort to show labels in order of confidence             
#            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            
            #NOTE: Effect of changing "predictions," to "predictions"
            predictions = sess.run(softmax_tensor, {input_layer_name: image_data})
            top_k = predictions[0].argsort()[-num_top_predictions:][::-1]
            
            print("Image File: {}, Top-{} Scores".format(test[i], \
                                      num_top_predictions))
            for node_id in top_k:
                predicted_label = labels[node_id]
#                score = predictions[node_id]
            #Effect of changing "predictions," to "predictions"
                score = predictions[0][node_id]
                print("Predicted Label: {}, Score: {}".format(predicted_label, \
                                              score))
                outfile.write(test[i] +', ' \
                              + predicted_label + ', ' \
                              + str(score) + '\n')
            i+=1
            print()


# In[8]:


src=os.path.join('.','test_img')
dest=os.path.join('.','test_img2')
labels='/home/rm/Sandlot-TensorFlow/Inception_wisdal/tmp/output_labels.txt'
graph='/home/rm/Sandlot-TensorFlow/Inception_wisdal/tmp/output_graph.pb'
input_layer='DecodeJpeg/contents:0'
output_layer='final_result:0'
num_top_predictions= 3    #1
labels = load_labels(labels)
load_graph(graph)
print("Running graph ...")
run_graph(src,dest,labels,\
          input_layer,\
          output_layer,\
          num_top_predictions)


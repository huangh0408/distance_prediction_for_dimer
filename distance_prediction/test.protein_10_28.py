import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *

n_epochs = 3000
learning_rate_val = 0.005
weight_decay_rate = 0.00001
momentum = 0.9
batch_size = 150
lambda_recon = 0.999
lambda_adv = 0.001

overlap_size = 0
hiding_size = 64

testset_path  = '../data/protein_testset_8_26.pickle'
result_path= '../results/test_10_28/'
dataset_path='../Study/data/protein_10_28'
pretrained_model_path = '../models/protein_10_28'

if not os.path.exists(result_path):
    os.makedirs( result_path )

if not os.path.exists( testset_path ):

    testset_dir = os.path.join( dataset_path, 'protein_eval_gt' )
    testset_chain_dir = os.path.join( dataset_path, 'protein_eval_chain' )
    testset_dir_temp=os.listdir(testset_dir)
    testset_dir_temp.sort()
    testset_chain_dir_temp=os.listdir(testset_chain_dir)
    testset_chain_dir_temp.sort()
    testset = pd.DataFrame({'image_path': map(lambda x: os.path.join( testset_dir, x ), testset_dir_temp),'chain_path':map(lambda x: os.path.join( testset_chain_dir, x ), testset_chain_dir_temp)})

    testset.to_pickle( testset_path )
else:
    testset = pd.read_pickle( testset_path )

is_train = tf.placeholder( tf.bool )
images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")

model = Model()

#reconstruction = model.build_reconstruction(images_tf, is_train)
bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction=model.build_reconstruction(images_tf, is_train)

# Applying bigger loss for overlapping region
#sess = tf.InteractiveSession()
#
#tf.initialize_all_variables().run()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
model_file=tf.train.latest_checkpoint(pretrained_model_path)
restorer = tf.train.Saver()
restorer.restore( sess, model_file )

#ii = 0
#for start,end in zip(
 #       range(0, len(testset), batch_size),
  #      range(batch_size, len(testset), batch_size)):
#print "%d" %ii
test_image_paths = testset[:batch_size]['image_path'].values
test_chain_paths = testset[:batch_size]['chain_path'].values
test_chain_ori=map(lambda x: load_chain( x ), test_chain_paths)
test_images_ori = map(lambda x: load_image(x), test_image_paths)

#    test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
test_images_crop = map(lambda x,y: crop_interaction(x, y), test_images_ori,test_chain_ori)
test_images, test_crops, xs,ys = zip(*test_images_crop)
bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, reconstruction_vals, recon_ori_vals = sess.run(
	[bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, reconstruction_ori,reconstruction],
        feed_dict={
        	images_tf: test_images,
#                images_hiding: test_crops,
                is_train: False
                })
#    reconstruction_vals = sess.run(
#            reconstruction,
#            feed_dict={
#                images_tf: test_images,
#                images_hiding: test_crops,
#                is_train: False
#                })
ii=0
for rec_val,img,x,y in zip(reconstruction_vals,test_images, xs, ys):
	rec_hid = (255. * (rec_val+1)/2.).astype(int)
        rec_con = (255. * (img+1)/2.).astype(int)
	rec_1=rec_con.copy()
        xx=128-x
        rec_2=misc.imresize(rec_hid[:,:,0],[y,xx],interp='nearest')
        rec_3=misc.imresize(rec_hid[:,:,1],[y,xx],interp='nearest')
        rec_4=misc.imresize(rec_hid[:,:,2],[y,xx],interp='nearest')
        rec_hid_temp=rec_1[0:y,x:128]
        rec_hid_temp[:,:,0]=rec_2
        rec_hid_temp[:,:,1]=rec_3
        rec_hid_temp[:,:,2]=rec_4
        rec_con[0:y, x:128] = rec_hid_temp
#        rec_con[y:y+64, x:x+64] = rec_hid
        #img_rgb = (255. * (img + 1)/2.).astype(int)
#        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.test.jpg'), rec_con)
	cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'_'+str(y)+'_'+str(xx)+'.test.jpg'), rec_con)
	cv2.imwrite( os.path.join(result_path, 'hid_img_'+str(ii)+'_'+str(y)+'_'+str(xx)+'.test.jpg'), rec_hid_temp)
        #cv2.imwrite( os.path.join(result_path, 'img_ori'+str(ii)+'.'+str(int(iters/1000))+'.jpg'), rec_con)
        ii += 1
#        if ii > 30: break


''' This code has only convolution during reading, but NO deconv. 
N = 12'''
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import os
import h5py

## MODEL PARAMETERS ##
model_file_name = "results/lstm_movie_maker"
C = 10
A,B = 64,64 # image width,height
img_size = B*A # the canvas size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
read_n = 12 # read glimpse grid width/height
write_n = 12 # write glimpse grid width/height
read_size = 2*read_n*read_n #if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n #if FLAGS.write_attn else img_size
z_size=10 # QSampler output size
T=10 # MNIST generation sequence length
batch_size=100 # training minibatch size
train_iters=10000
learning_rate=1e-3 # learning rate for optimizer
eps=1e-8 # epsilon for numerical stability
y_enc_size = 128

video_cell_size = 256

## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
# y = tf.placeholder(tf.int32,shape=(batch_size,10))
x_prev = tf.placeholder(tf.float32, shape=(batch_size, img_size))
#e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True) # decoder Op
video_cell = tf.contrib.rnn.LSTMCell(num_units=video_cell_size, state_is_tuple=True)

def next_batch(data_array):
	length=data_array.shape[0] #assuming the data array to be a np arry
	permutations=np.random.permutation(length)
	idxs=permutations[0:batch_size]
	batch=np.zeros([batch_size, C, img_size], dtype=np.float32)
	for i in range(len(idxs)):
		batch[i,:]=data_array[idxs[i]]
	return batch 


## definitions for convolution part of the code.

def fully_connected(bottom, n_out, name, reuse=DO_SHARE):
    shape = bottom.get_shape().as_list()
    with tf.variable_scope(name, reuse = reuse):
        # need to flattent he final result, find the dimension that is to be flattened
        dim = 1
        for d in shape[1:]:
            dim *= d
            # print(dim)
        x = tf.reshape(bottom, [-1, dim])
        
        weights = tf.get_variable('weights', [dim, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return tf.nn.relu(logits)


def deconv_layer(bottom, shape, output_shape, name, reuse = DO_SHARE): #doubtful about this
    with tf.variable_scope(name, reuse = reuse):
        # shape will be in the following form: [height, width, output_channels, input_channels]
        weights = tf.get_variable('weights', shape, tf.float32, xavier_initializer())
        biases = tf.get_variable('bias', shape[-2], tf.float32, tf.constant_initializer(0.0))
        dconv = tf.nn.conv2d_transpose(bottom, weights, output_shape = output_shape, strides = [1, 1, 1, 1], padding='VALID')
        activation = tf.nn.relu(tf.nn.bias_add(dconv, biases))
        # print(activation.get_shape())
        return activation

def conv_layer(bottom, shape, name, reuse = DO_SHARE):
    with tf.variable_scope(name, reuse = reuse):
    	# print 'hi'+name,reuse
        weights = tf.get_variable('weights', shape, tf.float32, xavier_initializer())
        biases = tf.get_variable('bias', shape[-1], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding = 'SAME')
        activation = tf.nn.relu(tf.nn.bias_add(conv, biases))
        return activation


def deconv_net(glimpse_enc, reuse = DO_SHARE):
    '''this network will take an input tensor of size (batchsize, glimpse_enc.shape) ''' 
    dfc_1 = fully_connected(glimpse_enc, 1024, 'dfc_1',reuse)
    dconv_1 = tf.reshape(dfc_1, [-1, 8, 8, 16])
    dconv_2 = deconv_layer(dconv_1, [3, 3, 8, 16], [batch_size, 10, 10, 8], 'deconv_2',reuse)
    dconv_3 = deconv_layer(dconv_2, [3, 3, 1, 8], [batch_size, 12, 12,1], 'deconv_3',reuse)
    # dconv_4 = deconv_layer(dconv_3, [5, 5, 1, 3], [batch_size, 28, 28,1], 'deconv_4')
    return dconv_3

def conv_net(input_tensor, reuse = DO_SHARE):
    ''' the function will take the input tensor (batchsize, A, B, Channels) and return the output of the first fully connected layer as the embedding for the VAE '''
    
    input_to_conv = tf.reshape(input_tensor,[batch_size, read_n, write_n, 1])
    n_in = input_to_conv.get_shape()[-1].value
    conv1_1 = conv_layer(input_to_conv, [3, 3, n_in, 8], "conv1_1", reuse)
    # pool1 = max_pool(conv1_1, 2, "pool1")

    conv2_1 = conv_layer(conv1_1, [3, 3, 8, 16], "conv2_1", reuse)
    # pool2 = max_pool(conv2_1, 2, "pool2")
    # return tf.reshape(conv2_1, [batch_size, 1025])
    fc1 = fully_connected(conv2_1, 512, 'fc1', reuse) 
    return fc1

def linear(x,output_dim):
	"""
	affine transformation Wx+b
	assumes x.shape = (batch_size, num_features)
	"""
	w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
	b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
	return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, N):
	grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
	mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
	mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
	a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
	b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
	mu_x = tf.reshape(mu_x, [-1, N, 1])
	mu_y = tf.reshape(mu_y, [-1, N, 1])
	sigma2 = tf.reshape(sigma2, [-1, 1, 1])
	Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
	Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
	# normalize, sum over A and B dims
	Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
	Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
	return Fx,Fy

def attn_window(scope,h_dec,N):
	with tf.variable_scope(scope,reuse=DO_SHARE):
		params=linear(h_dec,5)
	gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params, 5, axis = 1)
	gx=(A+1)/2*(gx_+1)
	gy=(B+1)/2*(gy_+1)
	sigma2=tf.exp(log_sigma2)
	delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
	return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(x,x_hat,h_dec_prev):
	return tf.concat([x,x_hat],1)

def read_attn_prev(x_prev,h_dec_prev):
	Fx,Fy,gamma=attn_window("read_prev",h_dec_prev,read_n)
	def filter_img(img,Fx,Fy,gamma,N):
		Fxt=tf.transpose(Fx,perm=[0,2,1])
		img=tf.reshape(img,[-1,B,A])
		glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
		glimpse=tf.reshape(glimpse,[-1,N*N])
		return glimpse*tf.reshape(gamma,[-1,1])
	x_prev=filter_img(x_prev,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
	return x_prev # concat along feature axis


def read_attn(x,x_hat,h_dec_prev):
	Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
	def filter_img(img,Fx,Fy,gamma,N, reuse = DO_SHARE):
		Fxt=tf.transpose(Fx,perm=[0,2,1])
		img=tf.reshape(img,[-1,B,A])
		glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))   
		conv_glimpse = conv_net(glimpse, reuse)
		# glimpse=tf.reshape(glimpse,[-1,N*N])
		return conv_glimpse*tf.reshape(gamma,[-1,1])
	x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
	x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n,True)
	return tf.concat([x,x_hat],1) # concat along feature axis


# def read_attn(x,x_hat,h_dec_prev):
# 	Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
# 	def filter_img(img,Fx,Fy,gamma,N):
# 		Fxt=tf.transpose(Fx,perm=[0,2,1])
# 		img=tf.reshape(img,[-1,B,A])
# 		glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
# 		glimpse=tf.reshape(glimpse,[-1,N*N])
# 		return glimpse*tf.reshape(gamma,[-1,1])
# 	x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
# 	x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
# 	return tf.concat([x,x_hat],1) # concat along feature axis

read = read_attn

def y_encode(y_d):
	with tf.variable_scope("y_enc"):
		y_enc = linear(y_d,y_enc_size)
		return y_enc

## ENCODE ## 
def encode(state,input):
	"""
	run LSTM
	state = previous encoder state
	input = cat(read,h_dec_prev)
	returns: (output, new_state)
	"""
	with tf.variable_scope("encoder",reuse=DO_SHARE):
		return lstm_enc(input,state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc,e):
	"""
	Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
	mu is (batch,z_size)
	"""
	with tf.variable_scope("mu",reuse=DO_SHARE):
		mu=linear(h_enc,z_size)
	with tf.variable_scope("sigma",reuse=DO_SHARE):
		logsigma=linear(h_enc,z_size)
		sigma=tf.exp(logsigma)
	return (mu + sigma*e, mu, logsigma, sigma,e)

## DECODER ## 
def decode(state,input):
	with tf.variable_scope("decoder",reuse=DO_SHARE):
		return lstm_dec(input, state)

def video_encode(state,input):
	with tf.variable_scope("video_encoder", reuse=DO_SHARE):
		return video_cell(input, state)

## WRITER ## 
def write_no_attn(h_dec):
	with tf.variable_scope("write",reuse=DO_SHARE):
		return linear(h_dec,img_size)

def write_attn(h_dec):
	with tf.variable_scope("writeW",reuse=DO_SHARE):
		w=linear(h_dec,write_size) # batch x (write_n*write_n)
	N=write_n
	w=tf.reshape(w,[batch_size,N,N])
	Fx,Fy,gamma=attn_window("write",h_dec,write_n)
	Fyt=tf.transpose(Fy,perm=[0,2,1])
	wr=tf.matmul(Fyt,tf.matmul(w,Fx))
	wr=tf.reshape(wr,[batch_size,B*A])
	#gamma=tf.tile(gamma,[1,B*A])
	return wr*tf.reshape(1.0/gamma,[-1,1])

write=write_attn

## STATE VARIABLES ## 

cs=[0]*T # sequence of canvases
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
et = [0]*T
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)


gif_train = tf.placeholder(tf.float32, shape = (batch_size, C, img_size))
gif_train_length = tf.placeholder(tf.int32, shape = (batch_size))
### LSTM Graph ###
gif_output, gif_state = tf.nn.dynamic_rnn(
    cell = video_cell,
    inputs = gif_train,
    dtype = tf.float32,
    sequence_length=gif_train_length,scope="video_encoder"
)

# outputs, states  = tf.nn.bidirectional_dynamic_rnn(
# 	cell_fw=video_cell,
# 	cell_bw=video_cell,
# 	dtype=tf.float32,
# 	sequence_length=gif_train_length,
# 	inputs=gif_train)

# h_gif = state
h_gif = tf.reshape(gif_output, [-1, C,video_cell_size])[:,-1]
print h_gif.get_shape()
## DRAW MODEL ##
# y_f = tf.cast(y, tf.float32)
# y_enc = y_encode(y_f)
# construct the unrolled computational graph
for t in range(T):
	c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
	x_hat=x-tf.sigmoid(c_prev) # error image
	r=read(x,x_hat,h_dec_prev)
	r_prev = read_attn_prev(x_prev,h_dec_prev)
	h_enc,enc_state=encode(enc_state,tf.concat([r,r_prev,h_dec_prev,h_gif],1))
	e=tf.random_normal((batch_size,z_size), mean=0, stddev=1)
	z,mus[t],logsigmas[t],sigmas[t],et[t]=sampleQ(h_enc,e)
	h_dec,dec_state=decode(dec_state,tf.concat([z, r_prev, h_gif],1))
	cs[t]=c_prev+write(h_dec) # store results
	h_dec_prev=h_dec
	DO_SHARE=True # from now on, share variables


cs_test = [0]*T
h_dec_test = [0] * T
dec_state_test = lstm_dec.zero_state(batch_size, tf.float32)
h_dec_prev_test = tf.zeros((batch_size,dec_size))

for t in range(T):
	r_prev_test = read_attn_prev(x_prev,h_dec_prev_test)
	e = tf.random_normal((batch_size,z_size), mean=0, stddev=1)
	c_prev_test = tf.zeros((batch_size,img_size)) if t==0 else cs_test[t-1]
	h_dec_test[t],dec_state_test = decode(dec_state_test,tf.concat([e,r_prev_test, h_gif], 1))
	cs_test[t] = c_prev_test+write(h_dec_test[t])
	h_dec_prev_test = h_dec_test[t]

h_gif_test,gif_state_test = video_encode(gif_state,tf.nn.sigmoid(cs_test[-1]))

## LOSS FUNCTION ## 

def binary_crossentropy(t,o):
	return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
	mu2=tf.square(mus[t])
	sigma2=tf.square(sigmas[t])
	logsigma=logsigmas[t]
	kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-T*.5 # each kl term is (1xminibatch)
KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx+Lz

## OPTIMIZER ## 

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
	if g is not None:
		grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

## RUN TRAINING ## 
dataset_file = 'dataset/mnist_single_gif.h5')

with h5py.File(dataset_file,'r') as hf:
	inputImages = np.float32(np.array(hf.get('mnist_gif_train')).reshape(-1,C,4096))
	#inputImages_val = np.float32(np.array(hf.get('mnist_gif_val')).reshape(-1,C,4096))

train_data = inputImages
#val_data = inputImages_val
# train_data = np.load('single_bouncing_mnist.npy')
print "loaded"

fetches=[]
fetches.extend([Lx,Lz,et,train_op])
Lxs=[0]*train_iters
Lzs=[0]*train_iters

sess=tf.InteractiveSession()

saver = tf.train.Saver() # saves variables learned during training

#saver.restore(sess, "/tmp/draw/drawmodel.ckpt") # to restore from model, uncomment this line
if os.path.isfile(model_file_name+".ckpt"):
	print("Restoring saved parameters")
	saver.restore(sess, model_file_name+".ckpt")
#     canvases,h_dec_ts = sess.run([cs_test,h_dec_test],feed_dict={})
#     canvases = np.array(canvases)
else:
	tf.initialize_all_variables().run()
	for i in range(train_iters):
		xtrain=next_batch(train_data) # xtrain is (batch_size x img_size)
		# xtrain = xtrain.reshape(-1,C,img_size) # xtrain is (batch_size x img_size)
		for j in range(C-1):
			if j==0:
				# print j
				seq_length = np.zeros((batch_size),dtype=np.int32)
				seq_length[:] = 1
				feed_dict={x:xtrain[:,j],x_prev:np.float32(np.zeros((batch_size,img_size))), gif_train_length: seq_length, gif_train: np.float32(np.zeros((batch_size, C, img_size)))}
			# elif: 
			# 	print j
			# 	seq_length = np.zeors((batch_size), dtype = np.int32)
			# 	seq_length[:] = j
			# 	gif_dummy = xtrain[:, : j-1]

			else:
				# print j
				seq_length = np.zeros((batch_size),dtype=np.int32)
				seq_length[:] = j
				# gif_train = np.zeros((batch_size, C, img_size))
				gif_dummy = xtrain[:, : j].reshape(batch_size, j, img_size)
				# print gif_dummy.shape
				dummy = np.zeros((batch_size, C - j, img_size))
				gif_train_1 = np.concatenate((gif_dummy, dummy), axis = 1)
				feed_dict={x:xtrain[:,j],x_prev:xtrain[:,j-1], gif_train_length: seq_length, gif_train: gif_train_1}
			results=sess.run(fetches,feed_dict)

		# feed_dict={x:xtrain,y:ytrain}
		# results=sess.run(fetches,feed_dict)
		Lxs[i],Lzs[i],et,_=results
		if i%10==0:
			print("iter=%d : Lx: %f Lz: %f" % (i,Lxs[i],Lzs[i]))
			#print np.array(et)[:,0,:]
		if (i+1)%500==0:
			ckpt_file=model_file_name+".ckpt"
			print("Model saved in file: %s" % saver.save(sess,ckpt_file))

	## TRAINING FINISHED ## 

	canvases=sess.run(cs,feed_dict) # generate some examples
	canvases=np.array(canvases) # T x batch x img_size

	out_file=model_file_name+".npy"
	np.save(out_file,[canvases,Lxs,Lzs])
	print("Outputs saved in file: %s" % out_file)

	ckpt_file=model_file_name+".ckpt"
	print("Model saved in file: %s" % saver.save(sess,ckpt_file))

	sess.close()

	print('Done drawing! Have a nice day! :)')

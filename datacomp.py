from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from sklearn import metrics
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
remote_flag=1
validation_rate=0.2
num_epoches=100
nb_classes=2
# Extract it into numpy arrays.
train_data = np.random.rand(10000,555)
train_labels=np.ones((10000,),dtype=np.int64)
test_x= np.random.rand(10000,555)
if remote_flag:
    test_y =np.ones((10000,),dtype=np.int64)

###############以上为读数据阶段###################
##################################################


train_data_size=train_data.shape[0]
VALIDATION_SIZE = np.floor(train_data_size*validation_rate).astype(np.int64)

feature_size=train_data.shape[1]
train_data=train_data.reshape(-1,feature_size,1)#shape=(batch,in_width,in_channels)


# Generate a validation set.
###############数据shuffle###########################
train_size = train_data.shape[0]
list_shuffle = list(range(train_size))
np.random.shuffle(list_shuffle)
train_data = train_data[list_shuffle]
train_labels = train_labels[list_shuffle]
#################################################
################训练数据分为validation和train####
validation_x = train_data[:VALIDATION_SIZE, ...]
validation_y = train_labels[:VALIDATION_SIZE]
train_x= train_data[VALIDATION_SIZE:,...]
train_y = train_labels[VALIDATION_SIZE:].reshape((-1,1))




####################以上为数据准备###########################
#############################################################

sess = tf.Session()
NUM_CHANNELS = 1
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
nb_classes = 2
# Extract it into numpy arrays.



def change_to_onehot(nb_classes, data):
    one_hot_data = np.eye(nb_classes)[data]
    return one_hot_data



######数据生成格式化
feature_size = train_x.shape[1]
test_feature_size = test_x.shape[1]
train_x = train_x.reshape(-1, feature_size, 1)  # shape=(batch,in_width,in_channels)
test_x = test_x.reshape(-1, test_feature_size, 1)

train_y = train_y.reshape((-1, 1))
train_y = change_to_onehot(nb_classes, train_y).reshape((-1, nb_classes)) #####train_y one_hot
validation_y = change_to_onehot(nb_classes, validation_y).reshape((-1, nb_classes)) #####validation_y one_hot

##############remote delete###################################33
if remote_flag:
    test_y = change_to_onehot(nb_classes, test_y).reshape((-1, nb_classes))


train_size = train_x.shape[0]
print(train_x.shape)
print(train_y.shape)
x = tf.placeholder(tf.float32, shape=[None, feature_size, 1])
y = tf.placeholder(tf.float32, shape=[None, nb_classes])

w1 = tf.Variable(tf.truncated_normal(shape=[3, 1, 32], stddev=0.1, dtype=tf.float32, name='w')) 
w2 = tf.Variable(tf.truncated_normal(shape=[3, 32, 64], stddev=0.1, dtype=tf.float32, name='w'))
conv1 = tf.nn.conv1d(x, w1, 2, 'VALID')  # 2为步长
print(conv1.shape)  # 宽度计算(width-kernel_size+1)/strides ,(10-3+1)/2=4  (64,4,32)
conv2 = tf.nn.conv1d(conv1, w2, 2, 'VALID')  # 步长为2
print(conv2.get_shape())  # 宽度计算(width-kernel_size+1)/strides ,(10-3+1)/2=4  (64,4,32)
conv2_shape = conv2.shape.as_list()
print(type(conv2_shape))
conv2_flatten = np.multiply(conv2_shape[1], 64)
fc1_W = tf.Variable(tf.truncated_normal([conv2_flatten, 1024], stddev=0.1, seed=SEED, dtype=tf.float32) / np.sqrt(conv2_flatten/2))
fc1_b = tf.Variable(tf.constant(0.1, shape=[1024]))
keep_prob = tf.placeholder("float")

fc2_W = tf.Variable(tf.truncated_normal([1024, nb_classes], stddev=0.1, seed=SEED, dtype=tf.float32)/np.sqrt(1024/2))
fc2_b = tf.Variable(tf.constant(0.1, shape=[nb_classes]))

conv2_shape = conv2.get_shape().as_list()
conv2_reshape = tf.reshape(conv2, [-1, conv2_shape[1] * conv2_shape[2]])

fc1 = tf.nn.relu(tf.matmul(conv2_reshape, fc1_W) + fc1_b)
fc1 = tf.nn.dropout(fc1, keep_prob, seed=SEED)
fc2 = tf.nn.softmax(tf.matmul(fc1, fc2_W) + fc2_b)
loss = -tf.reduce_mean(y * tf.log(fc2))
# L2 regularization for the fully connected parameters. ####添加 正则化项
regularizers = (tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc1_b) +
                tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc2_b)
                )
# Add the regularization term to the loss.
# loss += 5e-4 * regularizers
loss += 5e-4 * regularizers
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# train=optimizer.minimize(loss)
learning_rate = tf.train.exponential_decay(
    0.1,  # Base learning rate.
    100 * 64,  # Current index into the dataset.
    train_size,  # Decay step.
    0.1,  # Decay rate.
    staircase=True)
# train = tf.train.MomentumOptimizer(0.01,0.9).minimize(loss)
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

#####随机取train batch size
train_x_size = train_x.shape[0]
list_train_x_shuffle = list(range(train_x_size))
def next_batch(batch_size):
    begin = random.randint(0, train_x_size - batch_size)
    index = list_train_x_shuffle[begin:begin + batch_size]
    train_batch_x = train_x[index, ...]
    train_batch_y = train_y[index, :]
    return [train_batch_x, train_batch_y]

sess.run(tf.global_variables_initializer())
recall=0
precision=0
num_epochs=0
while !((recall>0.75) or (precision>0.75)):
    train_batch = next_batch(NUM_EPOCHS)
    train_batch_x= train_batch[0]
    train_batch_y = train_batch[1]
    _, l, pred = sess.run([train, loss, fc2], feed_dict={x: train_batch_x, y: train_batch_y, keep_prob: 0.3})
    print(num_epochs, l)
    if(num_epochs%100==0):
        validation_x_size = validation_x.shape[0]
        validation_predict_y = np.zeros(validation_x_size)
        for i in list(range(validation_x_size // 100)):
            min_index=min((i + 1) * 100,validation_x_size)
            validation_predict_x = validation_x[i * 100:min_index]
            validation_batch_y = validation_y[i * 100:(i + 1) * 100]

            validation_prediction = sess.run(fc2, feed_dict={x: validation_predict_x, y: validation_batch_y, keep_prob: 1})
            validation_prediction=np.argmax(validation_prediction, 1)
            validation_predict_y[i * 100:(i + 1) * 100] = validation_prediction
            print(i, validation_prediction.sum())
            validation_recall = metrics.recall_score(np.argmax(validation_y, 1), validation_predict_y, average='binary')
            validation_precision = metrics.precision_score(np.argmax(validation_y, 1), validation_predict_y, average='binary')
            print('validation_recall',validation_recall)
            print('validation_precision',validation_precision)
    num_epochs=num_epochs+1

test_x_size = test_x.shape[0]
predict_y = np.zeros(test_x_size)
num_epochs=0
for i in list(range(test_x_size // 100)):
    if num_epochs>10000:
        break
    min_index=min((i + 1) * 100,test_x_size)
    predict_x = test_x[i * 100:min_index]
    ####################3remote change####################3
    if remote_flag:
        test_batch_y = test_y[i * 100:(i + 1) * 100]
    else:
        test_batch_y = np.zeros([100, 2])
    prediction = sess.run(fc2, feed_dict={x: predict_x, y: test_batch_y, keep_prob: 1})
    result = np.array(prediction)
    pred = np.argmax(result, 1)

    predict_y[i * 100:(i + 1) * 100] = pred
    print(i, pred.sum())
print(predict_y.sum())

if remote_flag:
    recall=metrics.recall_score(np.argmax(test_y,1),predict_y,average='binary')
    precision=metrics.precision_score(np.argmax(test_y,1),predict_y,average='binary')
    f_score=metrics.f1_score(np.argmax(test_y,1),predict_y,average='binary')




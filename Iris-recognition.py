from __future__ import print_function
!pip install tensorflow
import tensorflow as tf
import os

MODE = 'folder'
DATASET_PATH = 'MMU/'
TEST_PATH = 'MMU2/'


N_CLASSES = 45 
IMG_HEIGHT = 24 
IMG_WIDTH = 32
CHANNELS = 3 

def read_images(dataset_path, mode, batch_size):
    imagepaths, labels = list(), list()
    if mode == 'file':
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        label = 0
        classes = None
        print(dataset_path)
        for (path, dirs, files) in os.walk(dataset_path):
            print("Dosyalar Okunuyor")
            print(path)
            classes = sorted(dirs)
            break
       
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            walk = os.walk(c_dir).__next__()
            
            for sample in walk[2]:
                if sample.endswith('.bmp'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

   
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
   
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    image = tf.read_file(image)
    image = tf.image.decode_bmp(image, channels=CHANNELS)

    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image * 1.0/127.5 - 1.0

    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=1)

    return X, Y

learning_rate = 0.0010
num_steps = 1000
batch_size = 64
display_step = 100

dropout = 0.75 


X, Y = read_images(DATASET_PATH, MODE, batch_size)
X2, Y2 = read_images(TEST_PATH, MODE, batch_size)

def conv_net(x, n_classes, dropout, reuse, is_training):
    
    with tf.variable_scope('ConvNet', reuse=reuse):

        
        conv1 = tf.layers.conv2d(x, 24, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 48, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 768)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_classes)
        out = tf.nn.softmax(out) if not is_training else out

    return out

logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
logits_test = conv_net(X2, N_CLASSES, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y2, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    tf.train.start_queue_runners()

    for step in range(1, num_steps+1):

        if step % display_step == 0:
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            sess.run(train_op)

    print("Optimization Finished!")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    coord.request_stop()
    coord.join(threads)

import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import time

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    train = pickle.load(f)

X, y = train['features'], train['labels']
    
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
x_resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

num_classes = 43
reg = 0.001
fc8_W_shape = (fc7.get_shape().as_list()[-1], num_classes)

# TODO: Add the final layer for traffic sign classification.
fc8_W = tf.get_variable('fc8_W', 
                        shape = fc8_W_shape,
                        initializer = tf.contrib.layers.xavier_initializer(uniform=False),
                        regularizer = tf.contrib.layers.l2_regularizer(reg))
fc8_b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32))
logits = tf.matmul(fc7, fc8_W) + fc8_b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, num_classes)

lrate = 0.001
cross_entropy = tf.losses.softmax_cross_entropy(one_hot_y, logits) + \
                tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = lrate)
training_operation = optimizer.minimize(loss_operation)
y_pred = tf.argmax(tf.nn.softmax(logits), 1)
labels = tf.argmax(one_hot_y, 1)
predictions = tf.argmax(logits, 1)
correct_prediction = tf.equal(predictions, labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
precision, precision_operation = tf.metrics.precision(labels, predictions)
recall, recall_operation = tf.metrics.recall(labels, predictions)
top_5_operation = tf.nn.top_k(probs, k=5)
saver = tf.train.Saver()

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset: offset + BATCH_SIZE], y_data[offset: offset + BATCH_SIZE]
        loss, accuracy, precision, recall = sess.run([loss_operation, 
                                                      accuracy_operation, 
                                                      precision_operation,
                                                      recall_operation], 
                                                 feed_dict={x: batch_x, y: batch_y})
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
        total_precision += (precision * len(batch_x))
        total_recall += (recall * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples, total_precision / num_examples, total_recall / num_examples

# TODO: Train and evaluate the feature extraction model.
EPOCHS = 50
BATCH_SIZE = 256
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
num_examples = len(X_train)
best_validation_accuracy = 0.0
#ts = str(int(time.time()))

#early_stopping = False
print("Training...")
print()
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    t0 = time.time()
    for batch_start in range(0, X_train.shape[0], BATCH_SIZE):
        batch_x = X_train[batch_start: batch_start + BATCH_SIZE]
        batch_y = y_train[batch_start: batch_start + BATCH_SIZE]
        _, loss, accuracy = sess.run([training_operation, loss_operation, accuracy_operation],
                                     feed_dict={x: batch_x, y: batch_y})
    
    #train_accuracy, train_loss, train_precision, train_recall = evaluate(X_train, y_train, sess)
    #print("Train")
    #print("  Acc = {:.3f}".format(train_accuracy))
    #print("  Loss = {:.3f}".format(train_loss))
    #print("  Precision = {:.3f}".format(train_precision))
    #print("  Recall = {:.3f}".format(train_recall))

    validation_accuracy, validation_loss, validation_precision, validation_recall = evaluate(X_valid, y_valid, sess)
    print("Epoch {} ...".format(i+1))
    print("  Time: %.3f seconds" % (time.time() - t0))
    print("Validation")
    print("  Acc = {:.3f}".format(validation_accuracy))
    print("  Loss = {:.3f}".format(validation_loss))
    print("  Precision = {:.3f}".format(validation_precision))
    print("  Recall = {:.3f}".format(validation_recall))

    # Save best model with val acc > 0.9
    #if validation_accuracy > 0.93 and validation_accuracy > best_validation_accuracy:
    #    best_validation_accuracy = validation_accuracy
    #    print("Saving model with new best validation accuracy {:.3f}".format(best_validation_accuracy))
    #    saver.save(sess, 'models/model-{}.chkp'.format(ts))
    #    early_stopping = True
        
    #if early_stopping:
    #    print('Done Training')
    #    break
    print()
print('Done training model model-{}.chkp'.format(ts))
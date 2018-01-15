import numpy as np
import tensorflow as tf
import pandas as pd

# data = pd.read_csv('train.csv').as_matrix()
# label = np.array(data[:,1], dtype=np.int32)
# label = np.eye(2)[label]

# d = pd.notnull(data[:,5])
# data = data[d]
# label = label[d]

def preprocess_data(path, is_test=False):
    data = pd.read_csv(path, index_col='PassengerId')
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    if is_test:
        data = data.replace([None], [0])
    else:
        data = data[pd.notnull(data['Age'])]
        data = data[pd.notnull(data['Embarked'])]
    data.replace(["female", "male"], [0, 1], inplace=True)
    data.replace(["Q", "C", "S"], [0, 1, 2], inplace=True)
    if "Survived" in data:
        data = data[pd.notnull(data['Survived'])]
    data_norm = (data - data.mean()) / (data.max() - data.min())
    return data_norm

dataset = preprocess_data("./train.csv")


label = pd.get_dummies(dataset.pop('Survived').values).as_matrix()
data = dataset.as_matrix()

x = tf.placeholder('float32', (None, data.shape[1]), name='x')
y = tf.placeholder('float32', (None, 2), name='y')


# W1 = tf.Variable(tf.random_normal([data.shape[1], 300]))
# W2 = tf.Variable(tf.random_normal([300, 2]))
# b1 = tf.Variable(tf.zeros([300]))
# b2 = tf.Variable(tf.zeros([2]))

# L1 = tf.sigmoid(tf.add(tf.matmul(x, W1), b1))
# logits = tf.sigmoid(tf.add(tf.matmul(L1, W2), b2))

with tf.name_scope('Model'):
    logits = tf.contrib.layers.fully_connected(x, 2)
# logits = tf.contrib.layers.fully_connected(logits, 2)


with tf.name_scope('Acc'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), 'float32'))
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
with tf.name_scope('GDS'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    log_writer = tf.summary.FileWriter(".", graph=tf.get_default_graph())

    history = []
    iterep = 500
    for i in range(iterep * 30):
        x_train = data[:700]
        y_train = label[:700]
        x_test = data[700:]
        y_test = label[700:]

        _, summary = sess.run([train_step, merged_summary], feed_dict={x: x_train, y: y_train})
        log_writer.add_summary(summary, i)
        # print(sess.run(logits, {x: [x_train[0]]}))
        if i % iterep == 0:
            tr = sess.run([loss, accuracy], feed_dict={x: x_train, y: y_train})
            t = sess.run([loss, accuracy], feed_dict={x: x_test, y: y_test})
            print(tr, t)

    test = preprocess_data("./test.csv", True)
    index = test.index.values
    test = test.as_matrix()
    predict_prob = logits.eval({x: test})
    predictions = tf.argmax(predict_prob, 1).eval()

    with open("kaggle.csv", "w") as f:
        f.write("PassengerId,Survived\n")
        for i, p in zip(index, predictions):
            f.write("{},{}\n".format(i, p))



    


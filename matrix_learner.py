import os

import tensorflow as tf
from matrix_learner_tf_records import cond_tf_record_parser, COND_FEATURE_LENGTH as INPUT_UNITS, parse_ast_data, TF_RECORDS_DIR, MATRICES_DIR


AST_INDEX = "ast"
CONDITION = "cond"

H1_UNITS = 256
H2_UNITS = 128
LEARNING_RATE = 1e-2
REGULARIZER_COEF = 0.1

BATCH_SIZE = 3
NUM_EPOCHS = 60
SHUFFLE_BUFFER_SIZE = 100

for root, dirs, files in os.walk(TF_RECORDS_DIR):
    for file in files:
        tf_record_path = os.path.join(TF_RECORDS_DIR, file)
        data_iter = tf.data.TFRecordDataset(tf_record_path)\
            .map(cond_tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
        preconds, postconds = data_iter.get_next()

        ast_to_id, asts = parse_ast_data()

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([INPUT_UNITS, H1_UNITS]), name="encoder_h1"),
            'linear_map': tf.Variable(tf.random_normal([H1_UNITS, H1_UNITS]), name='linear_map'),
            'decoder_h1': tf.Variable(tf.random_normal([H1_UNITS, INPUT_UNITS]), name='decoder_h1'),
        }

        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([H1_UNITS]), name='encoder_b1'),
            'decoder_b1': tf.Variable(tf.random_normal([INPUT_UNITS]), name='decoder_b1'),
        }


        def encoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            return layer_1


        def decoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
            return layer_1


        def linear_map(x):
            transform = tf.matmul(x, weights['linear_map'])
            return transform


        # Construct model
        P = tf.placeholder(tf.float32, [None, INPUT_UNITS])
        Q = tf.placeholder(tf.float32, [None, INPUT_UNITS])

        autoencoder_op = decoder(encoder(P))
        encoder_op = encoder(P)
        linear_op = linear_map(encoder_op)
        decoder_op = decoder(linear_op)

        P_true, P_pred = P, autoencoder_op
        Q_true, Q_pred = Q, decoder_op

        auto_loss = tf.reduce_mean(tf.pow(P_true - P_pred, 2))
        end_to_end_loss = tf.losses.sigmoid_cross_entropy(Q_true, Q_pred)
        regularizer = tf.nn.l2_loss(weights['encoder_h1']) \
                      + tf.nn.l2_loss(weights['decoder_h1']) \
                      + tf.nn.l2_loss(weights['linear_map'])
        loss = auto_loss + end_to_end_loss + REGULARIZER_COEF*regularizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(NUM_EPOCHS):
                sess.run(data_iter.initializer)
                while True:
                    try:
                        _, train_loss_val = sess.run([optimizer, loss], feed_dict={
                            P: sess.run(preconds),
                            Q: sess.run(postconds)
                        })
                        print('File: {} Epoch {}: Minibatch Loss: {}'.format(file, epoch, train_loss_val))
                    except tf.errors.OutOfRangeError:
                        break
                    except tf.errors.InvalidArgumentError:
                        break  # typically happens when there isn't enough data for a given AST
            ast_id, _ = os.path.splitext(os.path.basename(file))
            program_matrix_file = os.path.join(MATRICES_DIR, "{}.ckpt".format(ast_id))
            save_path = saver.save(sess, program_matrix_file)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # write matrix program matrix to file

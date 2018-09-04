import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from net_utils import create_dir, one_hot, imagen_grande
from dataset_utils import EMNIST
from AAEv2 import AAE
from sklearn.utils import shuffle
from OpenMax import OpenMax
from tqdm import tqdm
from generators import InputGenerator
import libmr
from scipy.spatial.distance import cdist

CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

def change_indexes(Y, classlist, new_indexes=None):
    """
    Change to values on new_indexes on specific class. Classlist must be of the same length of new_indexes, if new_indexes is None, will do it in order
    """
    if new_indexes is None:
        for idx, c in enumerate(classlist):
            Y[Y == c] = idx
    else:
        assert len(classlist) == len(new_indexes), "Must be of same size"
        for idx, c in zip(new_indexes, classlist):
            Y[Y == c] = idx
    return Y

def train(X, Y, args, increment=False, n_classes=10, exp=0, split=0, n_epochs=1):
    x_shape = [args.batch_size] + list(X.shape[1:])

    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None]+list(X.shape[1:]), n_classes, z_size=args.z_size, batch_size=args.batch_size, n_epochs=n_epochs)
        aae = AAE("train", batch_size=args.batch_size, n_epochs=n_epochs, n_classes=n_classes, z_size=args.z_size, input_shape=x_shape)

        iterador = dset.create_train_generator()
        (x_input, y_input), (z_real, y_real) = iterador.get_next()

        # Estructura
        z_hat, y_hat = aae.encoder(x_input)
        x_recon = aae.decoder(z_hat, y_hat)

        dz_real = aae.discriminator_z(z_real)
        dz_fake = aae.discriminator_z(z_hat)
        dy_real = aae.discriminator_y(y_input)
        dy_fake = aae.discriminator_y(y_hat)

        _, y_tilde = aae.encoder(x_input, supervised=True)

        # Metricas
        acc, acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(y_input, -1), tf.argmax(y_tilde, -1), n_classes)
        mse, mse_op = tf.metrics.mean_squared_error(x_input, x_recon)

        # Costos
        ae_loss = tf.losses.log_loss(x_input, x_recon)
        clf_loss = tf.losses.softmax_cross_entropy(y_input, y_tilde)

        dz_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dz_real), dz_real)
        dz_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dz_fake), dz_fake)
        dz_loss = dz_real_loss + dz_fake_loss

        dy_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dy_real), dy_real)
        dy_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dy_fake), dy_fake)
        dy_loss = dy_real_loss + dy_fake_loss

        gz_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dz_fake), dz_fake)
        gy_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dy_fake), dy_fake)
        gen_loss = gz_loss + gy_loss

        # Training ops
        all_vars = tf.trainable_variables()
        dz_vars = [var for var in all_vars if "Discriminator_Z" in var.name]
        dy_vars = [var for var in all_vars if "Discriminator_Y" in var.name]
        enc_vars = [var for var in all_vars if "Encoder" in var.name]
        ae_vars = enc_vars + [var for var in all_vars if "Decoder" in var.name]

        if increment:
            increment_vars = [var for var in tf.global_variables() if "Y_" not in var.name]
            increment_vars = [var for var in increment_vars if "Discriminator" not in var.name]
            init_vars = [var for var in tf.global_variables() if "Y_" in var.name]
            init_vars += [var for var in tf.global_variables() if "Discriminator" in var.name]
        else:
            increment_vars = None
            init_vars = None

        step_tensor = tf.Variable(0, trainable=False, name="Step")
        learning_rate = 0.001
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            ae_opt = tf.train.AdamOptimizer(learning_rate).minimize(ae_loss, var_list=ae_vars, global_step=step_tensor)
            dz_opt = tf.train.AdamOptimizer(learning_rate).minimize(dz_loss, var_list=dz_vars)
            dy_opt = tf.train.AdamOptimizer(learning_rate).minimize(dy_loss, var_list=dy_vars)
            gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=enc_vars)
            clf_opt = tf.train.AdamOptimizer(learning_rate).minimize(clf_loss, var_list=enc_vars)
            train_ops = tf.group([ae_opt,dz_opt,dy_opt,gen_opt,clf_opt])

        if increment:
            saver = tf.train.Saver(increment_vars)
        ckpt_saver = tf.train.Saver()

        with tf.Session() as sess:
            if increment:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
                # sess.run(tf.variables_initializer(init_vars))
            else:
                sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            #Cargar los datasets
            sess.run(iterador.initializer, feed_dict={dset.x_input:X, dset.y_input:Y})
            n_steps = (len(X) // args.batch_size)*n_epochs
            # Operacion de entrenamiento:
            with tqdm(desc="Train", total=n_steps, unit="Steps", miniters=10) as pbar:
                try:
                    while True:
                        _, step, accuracy, msqer, _, _ = sess.run([train_ops, step_tensor, acc, mse, acc_op, mse_op])
                        pbar.update()
                        if step % 10 == 0:
                            pbar.set_postfix(Accuracy=accuracy, MSE=msqer, refresh=False)
                except tf.errors.OutOfRangeError:
                    pass
            ckpt_saver.save(sess, save_path=save_file)
        print("Done!")

def eval(X, Y, args, n_classes=10, exp=0, split=0):
    x_shape = [args.batch_size] + list(X.shape[1:])

    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None]+list(X.shape[1:]), n_classes, args.z_size, batch_size=args.batch_size, n_epochs=1)
        aae = AAE("test", batch_size=args.batch_size, n_epochs=1, n_classes=n_classes, z_size=args.z_size, input_shape=x_shape)

        iterador = dset.create_eval_generator()
        x_input, y_input = iterador.get_next()
        _, y_tilde = aae.encoder(x_input)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(y_input, -1), tf.argmax(y_tilde, -1), n_classes)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
            sess.run(tf.local_variables_initializer())
            sess.run(iterador.initializer, feed_dict={dset.x_input:X, dset.y_input:Y})

            n_steps = (len(X) // args.batch_size)
            with tqdm(n_steps, desc="Eval", unit="Steps", leave=False) as pbar:
                try:
                    while True:
                        accuracy, _ = sess.run([acc, acc_op])
                        pbar.update()
                        pbar.set_postfix(Test_Acc=accuracy)
                except tf.errors.OutOfRangeError:
                    pass

            out_string = "Split {} Accuracy: {:02.3f}%".format(split + 1, 100 * accuracy)
            print(out_string)
            return out_string, accuracy

def get_train_fit(X, Y, args, n_classes=10, exp=0, split=0):
    x_shape = [args.batch_size] + list(X.shape[1:])

    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None]+list(X.shape[1:]), n_classes, args.z_size, batch_size=args.batch_size, n_epochs=1)
        aae = AAE("test", batch_size=args.batch_size, n_epochs=1, n_classes=n_classes, z_size=args.z_size, input_shape=x_shape)

        iterador = dset.create_eval_generator()
        x_input, y_input = iterador.get_next()
        _, y_tilde = aae.encoder(x_input, supervised=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
            sess.run(iterador.initializer, feed_dict={dset.x_input:X, dset.y_input:Y})

            n_steps = (len(X) // args.batch_size)
            with tqdm(n_steps, desc="Novelty Train", unit="Steps", leave=True) as pbar:
                try:
                    train_logits= np.empty([0, n_classes], dtype=np.float32)
                    while True:
                        logit = sess.run(y_tilde)
                        pbar.update()
                        train_logits = np.append(train_logits, logit, axis=0)
                except tf.errors.OutOfRangeError:
                    pass

            assert len(train_logits) == len(Y.argmax(-1))

            om = OpenMax(n_classes)
            try:
                om.fit_view(train_logits, Y.argmax(-1))
            except ValueError:
                print("No se pudo completar")
                return None
        return om

def get_novel_detections(X, Y, detector, args, n_classes=10, exp=0, split=0):
    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)
    with tf.Graph().as_default():
        dset = InputGenerator([None, 28, 28, 1], n_classes-1, args.z_size, batch_size=args.batch_size, n_epochs=1)
        aae = AAE("test", batch_size=args.batch_size, z_size=args.z_size, n_epochs=1, n_classes=n_classes-1)

        #modelo
        iterador = dset.create_test_generator()
        x_input = iterador.get_next()
        _, y_tilde = aae.encoder(x_input, supervised=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
            sess.run(iterador.initializer, {dset.x_input:X})
            eval_logits = np.empty((0, n_classes-1), dtype=np.float32)
            try:
                while True:
                    logit = sess.run(y_tilde)
                    eval_logits = np.append(eval_logits, logit, axis=0)
            except tf.errors.OutOfRangeError:
                pass

            detector.evaluate_view(eval_logits, Y)

         


def main(args):
    dataset = EMNIST()
    iniciales = np.arange(10)
    Xinit, Yinit = dataset.load_segment_of_data(iniciales, "train")
    Xinit, Yinit = shuffle(Xinit, Yinit, random_state=args.seed)

    #train(Xinit, one_hot(Yinit, 10), args, n_classes=10, exp=1, split=0, n_epochs=25)

    Xtest, Ytest = dataset.load_segment_of_data(iniciales, "test")
    Xtest = np.where(Xtest >= 0.5, 1.0, 0.0)
    eval_str, _ = eval(Xtest, one_hot(Ytest, 10), args, n_classes=10, exp=1, split=0)
    omax = get_train_fit(Xinit, one_hot(Yinit, 10), args, n_classes=10, exp=1, split=0)
    Xc, Yc = dataset.load_segment_of_data([10], kind="train")
    get_novel_detections(Xc, Yc, omax, args, n_classes=11, exp=1, split=0)
    print("Done!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default="extra/", help="Ruta de guardado del experimento y sus archivos")
    parser.add_argument("-e", "--n_exps", type=int, default=1, help="Numero de experimentos")
    parser.add_argument("--z_size", type=int, default=16, help="Tamaño de vector latente")
    parser.add_argument("--batch_size", type=int, default=32, help="Numero de batches")
    parser.add_argument("--n_epochs", type=int, default=10, help="Numero de epocas en entrenamiento inicial")
    parser.add_argument("--seed", type=int, default=348, help="Semilla a utilizar")
    parser.add_argument("--gen_threshold", type=float, default=0.75, help="Threshold para aceptar una imagen generada")
    parser.add_argument("--threshold", type=float, default=0.95, help="Threshold para rechazar input")
    parser.add_argument("--n_detects", type=float, default=0.05, help="Threshold de datos para considerar existencia de datos nuevos")
    args = parser.parse_args()
    main(args)

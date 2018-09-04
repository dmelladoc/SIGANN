import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from net_utils import create_dir, one_hot
from dataset_utils import EMNIST
from AAE import AAE
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from OpenMax import OpenMax
from tqdm import tqdm
from generators import InputGenerator

CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

def train(X, Y , args, n_classes=10, exp=0, split=0, n_epochs=1):
    x_shape = [args.batch_size] + list(X.shape[1:])
    y_shape = [args.batch_size] + list(Y.shape[1:])
    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None]+list(X.shape[1:]), n_classes, z_size=args.z_size, batch_size=args.batch_size, n_epochs=n_epochs)
        aae = AAE("train", batch_size=args.batch_size, n_epochs=n_epochs, z_size=args.z_size, n_classes=n_classes)

        iterador = dset.create_train_generator()
        (x_input, y_input), (z_real, y_real) = iterador.get_next()
        # Estructura
        z_hat, y_hat = aae.encoder(x_input)
        x_recon = aae.decoder(z_hat, y_hat)

        dz_real = aae.discriminator_z(z_real)
        dz_fake = aae.discriminator_z(z_hat)
        dy_real = aae.discriminator_y(y_real)
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

        step_tensor = tf.Variable(0, trainable=False, name="Step")
        learning_rate = 0.001
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            ae_opt = tf.train.AdamOptimizer(learning_rate).minimize(ae_loss, global_step=step_tensor)
            dz_opt = tf.train.AdamOptimizer(learning_rate).minimize(dz_loss, var_list=dz_vars)
            dy_opt = tf.train.AdamOptimizer(learning_rate).minimize(dy_loss, var_list=dy_vars)
            gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=enc_vars)
            clf_opt = tf.train.AdamOptimizer(learning_rate).minimize(clf_loss, var_list=enc_vars)
            train_ops = tf.group([ae_opt,dz_opt,dy_opt,gen_opt,clf_opt])

        ckpt_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

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
        print("Training Done!")


def eval(X, Y, args, n_classes=10, exp=0, split=0):
    x_shape = [args.batch_size] + list(X.shape[1:])
    y_shape = [args.batch_size] + list(Y.shape[1:])
    #save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

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
            with tqdm(n_steps, desc="Eval", unit="Steps") as pbar:
                try:
                    while True:
                        accuracy, _ = sess.run([acc, acc_op])
                        pbar.update()
                        pbar.set_postfix(Test_Acc=accuracy)
                except tf.errors.OutOfRangeError:
                    pass

            out_string = "Split {} Accuracy: {:02.3f}% \n".format(split + 1, 100 * accuracy)
            print(out_string)
            return out_string

def get_train_fit(X, Y, args, n_classes=10, exp=0, split=0):
    x_shape = [args.batch_size] + list(X.shape[1:])
    y_shape = [args.batch_size] + list(Y.shape[1:])
    #save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None]+list(X.shape[1:]), n_classes, args.z_size, batch_size=args.batch_size, n_epochs=1)
        aae = AAE("test", batch_size=args.batch_size, n_epochs=1, n_classes=n_classes, z_size=args.z_size, input_shape=x_shape)

        iterador = dset.create_eval_generator()
        x_input, y_input = iterador.get_next()
        _, y_tilde = aae.encoder(x_input, supervised=True)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(y_input, -1), tf.argmax(y_tilde, -1), n_classes)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
            sess.run(iterador.initializer, feed_dict={dset.x_input:X, dset.y_input:Y})

            n_steps = (len(X) // args.batch_size)
            with tqdm(n_steps, desc="Novelty Train", unit="Steps") as pbar:
                try:
                    train_logits= np.empty([0, n_classes], dtype=np.float32)
                    while True:
                        logit = sess.run(y_tilde)
                        pbar.update()
                        train_logits = np.append(train_logits, logit, axis=0)
                except tf.errors.OutOfRangeError:
                    pass

                except tf.errors.OutOfRangeError:
                    pass

            assert len(train_logits) == len(Y.argmax(-1))
            om = OpenMax(n_classes)
            om.fit(train_logits, Y.argmax(-1))

        return om


def main(args):
    dataset = EMNIST()

    with open("res_exp2.txt", "w") as f:
        for exp in np.arange(1, args.n_exps+1): #Haremos 1 validacion por ahora
            #exp=1
            f.write("\n-- Experimento {}/{} --\n".format(exp, args.n_exps))#1, 1))#

            #Directorios de guardado
            create_dir(args.save_path + "exp_{}/".format(exp))

            np.random.seed(args.seed)
            #Entrenaremos la red con los datos iniciales de los digitos
            Xtrain, Ytrain = dataset.load_segment_of_data(np.arange(10), "train")
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=args.seed)
            Ytrain = one_hot(Ytrain, 10)

            train(Xtrain, Ytrain, args, n_classes=10, exp=exp, n_epochs=args.n_epochs)
            #Evaluamos el desempeño del modelo
            Xtest, Ytest = dataset.load_segment_of_data(np.arange(10), "test")
            Ytest = one_hot(Ytest, 10)
            eval_str = eval(Xtest, Ytest, args, n_classes=10, exp=1)
            f.write(eval_str)
            f.write("Letra;SoftmaxAcc;OpenMaxAcc;Deteccion;Errores\n")

            omax = get_train_fit(Xtrain, Ytrain, args, n_classes=10, exp=1)
            save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, 0)

            Xtest, Ytest = dataset.load_segment_of_data(np.arange(10), "test")
            with tf.Graph().as_default():
                dset = InputGenerator([None,28,28,1], 10, args.z_size, batch_size=args.batch_size, n_epochs=1)
                aae = AAE("test", batch_size=args.batch_size, z_size=args.z_size, n_epochs=1, n_classes=10)
                iterador = dset.create_test_generator()
                x_input = iterador.get_next()
                _, y_tilde = aae.encoder(x_input, supervised=True)

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))

                    for i in np.arange(10, dataset.n_classes):
                        Xc, Yc = dataset.load_segment_of_data([i], kind="train")
                        Xchar = np.concatenate([Xtest, Xc], 0)
                        Ychar = np.concatenate([Ytest, Yc], 0)
                        Ychar[Ychar>9] = 10

                        sess.run(iterador.initializer, {dset.x_input:Xchar})
                        eval_logits = np.empty((0, 10), dtype=np.float32)
                        try:
                            while True:
                                logit = sess.run(y_tilde)
                                eval_logits = np.append(eval_logits, logit, axis=0)
                        except tf.errors.OutOfRangeError:
                            pass

                        openmax = omax.evaluate(eval_logits, Ychar)
                        softmax = omax._softmax(eval_logits)

                        Ypred = np.where(openmax.max(-1) <= args.threshold, 10, openmax.argmax(-1))
                        sm_acc = accuracy_score(Ychar, softmax.argmax(-1))
                        om_acc = accuracy_score(Ychar, Ypred)

                        detect = len(np.intersect1d(np.argwhere(Ypred == 10).squeeze(), np.argwhere(Ychar == 10).squeeze()))/len(Yc)
                        mistakes = len(np.intersect1d(np.argwhere(Ypred == 10).squeeze(), np.argwhere(Ychar < 10).squeeze()))/len(Yc)
                        res_text = "{};{:2.3f};{:2.3f};{:2.3f};{:2.3f}\n".format(CARACTERES[i], 100*sm_acc, 100*om_acc, 100*detect, 100*mistakes)
                        print(res_text)
                        f.write(res_text)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["EMNIST", "CIFAR"], type=str, help="Dataset a evaluar")
    parser.add_argument("-s", "--save_path", type=str, default="exp2/", help="Ruta de guardado del experimento y sus archivos")
    parser.add_argument("-e", "--n_exps", type=int, default=1, help="Numero de experimentos")
    parser.add_argument("--z_size", type=int, default=16, help="Tamaño de vector latente")
    parser.add_argument("--batch_size", type=int, default=32, help="Numero de batches")
    parser.add_argument("--n_epochs", type=int, default=10, help="Numero de epocas en entrenamiento inicial")
    parser.add_argument("--seed", type=int, default=348, help="Semilla a utilizar")
    parser.add_argument("--n_generate", type=int, default=500, help="Numero de semillas generadas")
    parser.add_argument("--threshold", type=float, default=0.95, help="Threshold para rechazar input")
    args = parser.parse_args()

    main(args)

import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from net_utils import create_dir, one_hot, imagen_grande
from dataset_utils import EMNIST
from AAE import AAE
from sklearn.utils import shuffle
from OpenMax import OpenMax
from tqdm import tqdm
from generators import InputGenerator

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
                om.fit(train_logits, Y.argmax(-1))
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

            openmax = detector.evaluate(eval_logits, Y)

            Ypred = np.where(openmax.max(-1) <= args.threshold, n_classes, openmax.argmax(-1))

            detect = np.intersect1d(np.argwhere(Ypred == n_classes).squeeze(), np.argwhere(Y == n_classes-1).squeeze()) #detectar los casos correctos
            percentage = len(detect)/len(Y[Y == (n_classes-1)])
            res_text = "D={} ({:.2f}%)\n".format(len(detect), 100*percentage)
            print(res_text)

            if percentage <= args.n_detects:
                print("No hay realmente información nueva")
                return False, None, None, res_text
            else:
                return True, X[detect], Y[detect], res_text

def generate(args, n_classes=10, exp=0, split=0, n_generados=1000):
    #x_shape = [args.batch_size] + list(X.shape[1:])
    #y_shape = [args.batch_size] + list(Y.shape[1:])
    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None, 28, 28, 1], n_classes, args.z_size, batch_size=args.batch_size, n_epochs=1)
        aae = AAE("test", batch_size=n_classes, n_epochs=1, n_classes=n_classes, z_size=args.z_size, input_shape=[None,28,28,1])

        iterador = dset.create_gen_generator()
        z_input, y_input = iterador.get_next()

        x_recon = aae.decoder(z_input, y_input) #neutralizar los numeros
        #x_bin = tf.cast(tf.greater_equal(x_recon, 0.5), tf.float32) #binarizar lo generado
        _, y_tilde = aae.encoder(x_recon)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(y_input, -1), tf.argmax(y_tilde, -1), n_classes)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
            sess.run(tf.local_variables_initializer())
            sess.run(iterador.initializer)

            X_gen = np.empty((0, 28, 28, 1), dtype=np.float32)
            Y_gen = np.empty((0), dtype=np.int32)

            with tqdm(desc="Generador", unit="Steps") as pbar:
                while True:
                    x_p, y_p, y_clases, accuracy, _ = sess.run([x_recon, y_tilde, y_input, acc, acc_op])
                    pbar.update()
                    idx = (y_clases.argmax(-1) == y_p.argmax(-1)) & (y_p.max(-1) >= args.gen_threshold)
                    X_gen = np.append(X_gen, x_p[idx], axis=0)
                    Y_gen = np.append(Y_gen, y_p[idx].argmax(-1), axis=0)
                    n_gen = np.bincount(Y_gen).mean()
                    pbar.set_postfix(Gen_acc=accuracy, D_gen=n_gen, refresh=False)
                    if n_gen >= n_generados:
                        break

            assert len(X_gen) == len(Y_gen)
            gen_str = "Generados:" + str(np.bincount(Y_gen)) + "\n"
            print(gen_str)
            # print("Elementos Generados por Clase:", np.bincount(Y_gen))
            return X_gen, Y_gen, gen_str

def main(args):
    if args.dataset == "MNIST":
        dataset = MNIST()
    elif args.dataset == "CIFAR10":
        dataset = CIFAR10()
    elif args.dataset == "EMNIST":
        dataset = EMNIST()
    else:  # args.dataset == "EMNIST":
        raise NotImplementedError("Aun no está disponible")

    with open("res_exp3.txt", "w") as f:
        for exp in np.arange(1, args.n_exps +1):
            f.write("\n-- Experimento {}/{} --\n".format(exp, args.n_exps))
            n_epochs = args.n_epochs
            #Directorios de guardado
            create_dir(args.save_path + "exp_{}/".format(exp))
            img_dir = args.save_path + "imgs/img_{}/".format(exp)
            create_dir(img_dir)
            #np.random.seed(args.seed)
            #clases indexadas
            #real_classes = np.arange(dataset.n_classes)
            if args.ordered:
                shuffled_classes = np.append(np.arange(args.starting),np.random.permutation(np.arange(args.starting, dataset.n_classes))) #Si lo hacemos ordenado
            else:
                shuffled_classes = np.random.permutation(dataset.n_classes)
            initial_classes = shuffled_classes[:args.starting]
            print("Clases Iniciales: " + ",".join([CARACTERES[ch] for ch in initial_classes]))
            unknown_classes = shuffled_classes[args.starting:]

            #Cargamos las clases iniciales
            Xinit, Yinit = dataset.load_segment_of_data(initial_classes, "train")
            Yinit = change_indexes(Yinit, initial_classes) #cambiamos del indice de 0- starting
            Xinit, Yinit = shuffle(Xinit, Yinit, random_state=args.seed)
            #Xinit = np.where(Xinit >= 0.5, 1.0, 0.0) #binarizar

            train(Xinit, one_hot(Yinit, args.starting), args, n_classes=args.starting, exp=exp, split=0, n_epochs=n_epochs)

            #Evaluamos el desempeño inicial
            Xitest, Yitest = dataset.load_segment_of_data(initial_classes, "test")
            Yitest = change_indexes(Yitest, initial_classes)
            Xitest = np.where(Xitest >= 0.5, 1.0, 0.0) #binarizar
            eval_str, _ = eval(Xitest, one_hot(Yitest, args.starting), args, n_classes=args.starting, exp=exp, split=0)
            f.write("--" + eval_str)

            #Creamos nuestra instancia de OpenMax
            try:
                omax = get_train_fit(Xinit, one_hot(Yinit, args.starting), args, n_classes=args.starting, exp=exp, split=0)
                if omax is None:
                    print("Muy pocos casos")
                    f.write("\nno se pudo ajustar mas\n---END---\n")
                    continue
            except ValueError:
                continue


            for idx, unk in enumerate(unknown_classes):
                print("Reconociendo " + CARACTERES[unknown_classes[idx]])
                n_epochs = 5 + int(args.n_epochs*(0.9**idx)) if args.decay_epoch else int(args.n_epochs / 2)#.astype(np.int) #in(0.95*n_epochs) if n_epochs > 8 else 10
                #if not is_increment:
                #    Xitest, Yitest = dataset.load_segment_of_data(initial_classes)
                #else:
                initial_classes = shuffled_classes[:args.starting + idx]
                Xitest, Yitest = dataset.load_segment_of_data(initial_classes, kind="test")
                Xc, Yc = dataset.load_segment_of_data([unk], kind="train")

                Xnew = np.concatenate([Xitest, Xc], 0)
                Ynew = np.concatenate([Yitest, Yc], 0)
                nclasses = args.starting + idx + 1
                Ynew = change_indexes(Ynew, shuffled_classes[:nclasses])
                #Xnew = np.where(Xnew >= 0.5, 1.0, 0.0) #binarizar

                is_unks, Xunk, Yunk, res_text = get_novel_detections(Xnew, Ynew, omax, args, n_classes=nclasses, exp=exp, split=idx)
                #Revisamos si hay casos nuevos o no:
                f.write(res_text)
                if not is_unks:
                    print("---END---")
                    f.write("---END---\n")
                    break

                #Generamos datos
                Xgen, Ygen, gen_text = generate(args, n_classes=nclasses-1, exp=exp, split=idx, n_generados=int(0.9*len(Yunk)))
                imagen_grande(Xgen, 10, img_dir + "recuerdo{:02d}.png".format(idx))
                #Los unimos con los datos nuevos
                Xinit = np.concatenate([Xgen, Xunk], axis=0)
                Yinit = np.concatenate([Ygen, Yunk], axis=0)
                Xinit, Yinit = shuffle(Xinit, Yinit, random_state=args.seed)
                #Xinit = np.where(Xinit >= 0.5, 1.0, 0.0) #binarizar

                #Entrenamos
                train(Xinit, one_hot(Yinit, nclasses), args, n_classes=nclasses, increment=True, exp=exp, split=idx+1, n_epochs=n_epochs)#int(0.8*args.n_epochs))

                #Evaluamos
                Xitest, Yitest = dataset.load_segment_of_data(shuffled_classes[:nclasses], "test")
                Yitest = change_indexes(Yitest, shuffled_classes[:nclasses])
                #Xitest = np.where(Xitest >= 0.5, 1.0, 0.0) #binarizar
                eval_str, acc = eval(Xitest, one_hot(Yitest, nclasses), args, n_classes=nclasses, exp=exp, split=idx+1)
                f.write(CARACTERES[unknown_classes[idx]] + "-" + eval_str)

                #Caso de olvido catastrofico
                olvido = (1/nclasses) + 0.05
                if acc <= olvido:
                    forget_text = "\nAccuracy {:.3f} <= {:.3f}.\n---Olvido Catastrofico---\n".format(100*acc, 100*olvido)
                    print(forget_text)
                    f.write(forget_text)
                    break

                #Y creamos nuestro nuevo detector
                omax = get_train_fit(Xinit, one_hot(Yinit, nclasses), args, n_classes=nclasses, exp=exp, split=idx+1)
                if omax is None:
                    print("Muy pocos casos")
                    f.write("\nno se pudo ajustar mas\n---END---\n")
                    break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["EMNIST", "CIFAR"], type=str, help="Dataset a evaluar")
    parser.add_argument("-s", "--save_path", type=str, default="exp03/", help="Ruta de guardado del experimento y sus archivos")
    parser.add_argument("-e", "--n_exps", type=int, default=1, help="Numero de experimentos")
    parser.add_argument("-n", "--starting", type=int, default=5, help="Numero de clases iniciales")
    parser.add_argument("--z_size", type=int, default=16, help="Tamaño de vector latente")
    parser.add_argument("--batch_size", type=int, default=32, help="Numero de batches")
    parser.add_argument("--n_epochs", type=int, default=10, help="Numero de epocas en entrenamiento inicial")
    parser.add_argument("--seed", type=int, default=348, help="Semilla a utilizar")
    parser.add_argument("--gen_threshold", type=float, default=0.75, help="Threshold para aceptar una imagen generada")
    parser.add_argument("--threshold", type=float, default=0.95, help="Threshold para rechazar input")
    parser.add_argument("--n_detects", type=float, default=0.05, help="Threshold de datos para considerar existencia de datos nuevos")
    parser.add_argument("--decay_epoch", action="store_true", help="Decaer el epoch")
    parser.add_argument("--ordered", action="store_true", help="Clases Ordenadas")
    args = parser.parse_args()
    main(args)

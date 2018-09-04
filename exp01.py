import os
import argparse
import numpy as np
import tensorflow as tf
from net_utils import create_dir, one_hot, imagen_grande
from dataset_utils import MNIST, CIFAR10, EMNIST
from AAE import AAE
from sklearn.utils import shuffle
from tqdm import tqdm, trange
from generators import InputGenerator


def train(X, Y, args, increment=False, n_classes=10, exp=0, split=0, n_epochs=1):
    x_shape = [args.batch_size] + list(X.shape[1:])
    y_shape = [args.batch_size] + list(Y.shape[1:])
    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)
    monitor_path = args.monitor + "exp_{}/split_{}".format(exp, split)

    with tf.Graph().as_default():

        dset = InputGenerator([None]+list(X.shape[1:]), n_classes=n_classes, args.z_size, batch_size=args.batch_size, n_epochs=n_epochs)
        aae = AAE("train", batch_size=args.batch_size, n_epochs=n_epochs, n_classes=n_classes, z_size=args.z_size, input_shape=x_shape)

        iterador = dset.create_train_generator()
        (x_input, y_input), (z_real, y_real) = iterador.get_next()
        # Estructura
        z_hat, y_hat = aae.encoder(x_input)

        pz = tf.sigmoid(z_hat + 1e-8)
        entropia = tf.reduce_mean(-tf.reduce_sum(pz * tf.log(pz),1))
        #entropia = tf.reduce_mean(0.5*tf.norm(z_hat - z_real, ord=1, axis=1))
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
        ae_loss = tf.losses.log_loss(x_input, x_recon) # tf.reduce_mean(aae.binary_crossentropy(x_input, x_recon))
        clf_loss = tf.losses.softmax_cross_entropy(y_input, y_tilde) #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_tilde))

        dz_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dz_real), dz_real) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dz_real), logits=dz_real))
        dz_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dz_fake), dz_fake) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dz_fake), logits=dz_fake))
        dz_loss = dz_real_loss + dz_fake_loss

        dy_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dy_real), dy_real) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dy_real), logits=dy_real))
        dy_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dy_fake), dy_fake) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dy_fake), logits=dy_fake))
        dy_loss = dy_real_loss + dy_fake_loss

        gz_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dz_fake), dz_fake) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dz_fake), logits=dz_fake))
        gy_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dy_fake), dy_fake) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dy_fake), logits=dy_fake))
        gen_loss = gz_loss + gy_loss

        # Training ops
        all_vars = tf.trainable_variables()
        dz_vars = [var for var in all_vars if "Discriminator_Z" in var.name]
        dy_vars = [var for var in all_vars if "Discriminator_Y" in var.name]
        enc_vars = [var for var in all_vars if "Encoder" in var.name]

        if increment:
            increment_vars = [var for var in tf.global_variables() if "Y_" not in var.name]
            init_vars = [var for var in tf.global_variables() if "Y_" in var.name]
        else:
            increment_vars = None
            init_vars = None

        step_tensor = tf.Variable(0, trainable=False, name="Step")
        #learning_rate = tf.train.polynomial_decay(0.005, global_step=step_tensor, decay_steps=20000, end_learning_rate=0.000001, power=2, name="Learning_rate")
        learning_rate = 0.001
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            ae_opt = tf.train.AdamOptimizer(learning_rate).minimize(ae_loss, global_step=step_tensor)
            dz_opt = tf.train.AdamOptimizer(learning_rate).minimize(dz_loss, var_list=dz_vars)
            dy_opt = tf.train.AdamOptimizer(learning_rate).minimize(dy_loss, var_list=dy_vars)
            gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=enc_vars)
            clf_opt = tf.train.AdamOptimizer(learning_rate).minimize(clf_loss, var_list=enc_vars)
            train_ops = tf.group([ae_opt,dz_opt,dy_opt,gen_opt,clf_opt])

        # summaries
        tf.summary.scalar("Losses/AE_loss", ae_loss)
        tf.summary.scalar("Losses/Dis_Y_loss", dy_loss)
        tf.summary.scalar("Losses/Dis_Z_loss", dz_loss)
        tf.summary.scalar("Losses/Gen_loss", gen_loss)
        tf.summary.scalar("Losses/Clf_loss", clf_loss)
        tf.summary.scalar("Metrics/Accuracy", acc)
        tf.summary.scalar("Metrics/MSE", mse)
        tf.summary.scalar("Metrics/Entropy", entropia)
        #tf.summary.scalar("Metrics/LearningRate",learning_rate)
        tf.summary.histogram("Z_pred", z_hat)
        tf.summary.histogram("Z_real", z_real)
        tf.summary.image("X_Real", x_input, 10)
        tf.summary.image("X_Recon", x_recon, 10)
        summary_op = tf.summary.merge_all()

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
            summary_writer = tf.summary.FileWriter(monitor_path)
            #Cargar los datasets
            sess.run(iterador.initializer, feed_dict={dset.x_input:X, dset.y_input:Y})
            n_steps = (len(X) // args.batch_size)*n_epochs
            # Operacion de entrenamiento:
            with tqdm(desc="Train", total=n_steps, unit="Steps", miniters=10) as pbar:
                try:
                    while True:
                        _, step, accuracy, msqer, _, _, summary = sess.run([train_ops, step_tensor, acc, mse, acc_op, mse_op, summary_op])
                        summary_writer.add_summary(summary, step)
                        pbar.update()
                        if step % 10 == 0:
                            pbar.set_postfix(Accuracy=accuracy, MSE=msqer, refresh=False)

                except tf.errors.OutOfRangeError:
                    pass
            ckpt_saver.save(sess, save_path=save_file)
        print("Done!")


def eval(X, Y, args, n_classes=10, exp=0, split=0):
    x_shape = [args.batch_size] + list(X.shape[1:])
    y_shape = [args.batch_size] + list(Y.shape[1:])
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
            with tqdm(n_steps, desc="Eval", unit="Steps") as pbar:
                try:
                    while True:
                        accuracy, _ = sess.run([acc, acc_op])
                        pbar.update()
                        pbar.set_postfix(Test_Acc=accuracy, refresh=False)
                except tf.errors.OutOfRangeError:
                    pass

            out_string = "Split {} Accuracy: {:02.3f}% ".format(split + 1, 100 * accuracy)
            print(out_string)
            return out_string
            # print("Split {} Accuracy: {:02.3f}%".format(split+1, 100 * accuracy))


def generate(X, Y, args, n_classes=10, exp=0, split=0, n_generados=1000):
    x_shape = [args.batch_size] + list(X.shape[1:])
    y_shape = [args.batch_size] + list(Y.shape[1:])
    save_file = args.save_path + "exp_{}/model_split{}.ckpt".format(exp, split)

    with tf.Graph().as_default():
        dset = InputGenerator([None]+list(X.shape[1:]), n_classes, args.z_size, batch_size=args.batch_size, n_epochs=1)
        aae = AAE("test", batch_size=n_classes, n_epochs=1, n_classes=n_classes, z_size=args.z_size, input_shape=x_shape)

        iterador = dset.create_gen_generator()
        z_input, y_input = iterador.get_next()

        x_recon = aae.decoder(z_input, y_input) #neutralizar los numeros
        _, y_tilde = aae.encoder(x_recon)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(y_input, -1), tf.argmax(y_tilde, -1), n_classes)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_path + "exp_{}/".format(exp)))
            sess.run(tf.local_variables_initializer())
            sess.run(iterador.initializer)

            X_gen = np.empty((0,28,28,1), dtype=np.float32)#[]
            Y_gen = np.empty((0,n_classes), dtype=np.float32)#[]
            with tqdm(desc="Generador", unit="Steps") as pbar:
                while True:#for step in pbar:
                    #y_clases = one_hot(np.random.randint(n_classes, size=args.batch_size), n_classes)
                    #Z = np.random.normal(0, 0.75, (args.batch_size, args.z_size))
                    x_p, y_p, y_clases, accuracy, _ = sess.run([x_recon, y_tilde, y_input, acc, acc_op])

                    pbar.update()
                    idx = (y_clases.argmax(-1) == y_p.argmax(-1))# & (y_p.max(-1) >= 0.9)
                    # idx = np.argwhere(y_clases.argmax(-1) == y_p.argmax(-1)).squeeze()
                    X_gen = np.append(X_gen, x_p[idx], axis=0)#+= [x_p[idx]]
                    Y_gen = np.append(Y_gen, y_p[idx], axis=0)#+= [y_p[idx]]
                    ngen = np.bincount(Y_gen.argmax(-1)).mean()
                    pbar.set_postfix(Gen_Acc=accuracy, D_Gen=ngen, refresh=False)
                    if ngen >= n_generados:
                        break


            #print("Generated Acc={:02.3f}%".format(100*accuracy))
            assert len(X_gen) == len(Y_gen)
            #X_gen = np.concatenate(X_gen, axis=0)
            #Y_gen = np.concatenate(Y_gen, axis=0).argmax(-1)
            Y_gen = Y_gen.argmax(-1)

            gen_str = "Elementos:" + str(np.bincount(Y_gen)) + "\n"
            print(gen_str)
            # print("Elementos Generados por Clase:", np.bincount(Y_gen))
            return X_gen, Y_gen, gen_str


def main(args):
    # Cargar dataset seleccionado
    if args.dataset == "MNIST":
        dataset = MNIST()
    elif args.dataset == "CIFAR10":
        dataset = CIFAR10()
    elif args.dataset == "EMNIST":
        dataset = EMNIST()
    else:  # args.dataset == "EMNIST":
        raise NotImplementedError("Aun no está disponible")

    assert dataset.n_classes > args.segments, "Segmentos deben ser menos que las clases"
    f = open("resultados.txt", "w")

    for exp in np.arange(1, args.n_exps + 1):
        f.write("\n-- Experimento {}/{} --\n".format(exp, args.n_exps))
        # print("-- Experimento {}/{} --".format(exp, args.n_exps))
        create_dir(args.save_path + "exp_{}/".format(exp))
        create_dir(args.monitor + "exp_{}/".format(exp))  # Creamos los directorios de guardado
        create_dir("gen_img/exp_{}/".format(exp))

        np.random.seed(args.seed)

        # Seleccionamos los segmentos de datos
        splits = np.array_split(np.arange(dataset.n_classes), args.segments)
        for split_idx, split in enumerate(splits):
            f.write("Split {}: ".format(split_idx + 1) + str(split) + "\n")
            # print("Split {}".format(split_idx + 1), split)
            split_classes = split.max() + 1  # Clases para empezar la etapa incremental

            # Si es la primera etapa:
            if split_idx == 0:
                train_epochs = args.n_epochs
                is_increment = False
                X_in, Y_in = dataset.load_segment_of_data(split, "train")
            else:
                train_epochs = int(0.75 * train_epochs)#args.n_epochs / 2)  # 5 #Muy pocos epochs de entrenamiento al parecer
                is_increment = True
                _X_in, _Y_in = dataset.load_segment_of_data(split, "train")
                if args.rehearsal:
                    X_r, Y_r = dataset.load_percent_of_data(np.concatenate(splits[:split_idx], axis=0), args.percentage, "train")
                    X_in = np.concatenate([_X_in, X_r, X_pseudo], axis=0)
                    Y_in = np.concatenate([_Y_in, Y_r, Y_pseudo], axis=0)
                else:
                    X_in = np.concatenate([_X_in, X_pseudo], axis=0)
                    Y_in = np.concatenate([_Y_in, Y_pseudo], axis=0)

            X_in, Y_in = shuffle(X_in, Y_in, random_state=args.seed)
            X_in = np.where(X_in >= 0.5, 1.0, 0.0) #binarizar

            Y_in = one_hot(Y_in, split_classes)
            train(X_in, Y_in, args, increment=is_increment, n_classes=split_classes, exp=exp, split=split_idx, n_epochs=train_epochs)
            # Evaluamos el accuracy

            X_test, Y_test = dataset.load_segment_of_data(np.arange(split_classes), "test")
            X_test = np.where(X_test >= 0.5, 1.0, 0.0) #binarizar
            Y_test = one_hot(Y_test, split_classes)

            eval_str = eval(X_test, Y_test, args, n_classes=split_classes, exp=exp, split=split_idx)
            f.write(eval_str)

            X_pseudo, Y_pseudo, gen_str = generate(X_test, Y_test, args, n_classes=split_classes, exp=exp, split=split_idx, n_generados=args.n_generate)
            f.write(gen_str)

            gen_img_file = "gen_img/exp_{}/generado_e{:02d}_s{:02d}.png".format(exp, exp, split_idx + 1)
            imagen_grande(X_pseudo[:100], n=10, out_filename=gen_img_file)

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["MNIST", "EMNIST", "CIFAR10"], type=str, help="Dataset a evaluar")
    parser.add_argument("segments", type=int, help="Numero de segmentos")
    parser.add_argument("-s", "--save_path", type=str, default="ckpt/", help="Ruta de guardado")
    parser.add_argument("-m", "--monitor", type=str, default="monitor/", help="Ruta de monitor para Tensorboard")
    parser.add_argument("-e", "--n_exps", type=int, default=1, help="Numero de experimentos")
    parser.add_argument("--z_size", type=int, default=10, help="Tamaño de vector latente")
    parser.add_argument("--batch_size", type=int, default=32, help="Numero de batches")
    parser.add_argument("--n_epochs", type=int, default=10, help="Numero de epocas en entrenamiento inicial")
    parser.add_argument("--seed", type=int, default=348, help="Semilla a utilizar")
    parser.add_argument("--n_generate", type=int, default=100, help="Numero de semillas generadas")
    parser.add_argument("--rehearsal", action="store_true", help="Añadir Rehearsal")
    parser.add_argument("--percentage", type=float, default=0.5, help="Porcentaje de datos en rehearsal")
    args = parser.parse_args()

    main(args)

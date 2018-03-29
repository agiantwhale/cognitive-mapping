from threading import Thread, Lock, Condition
from Queue import PriorityQueue, Empty
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
from expert import Expert
from model import CMAP
from copy import deepcopy
import argparse
import random
import cv2
import time

FLAGS = None


class PriorityLock(object):
    def __init__(self):
        self._is_available = True
        self._mutex = Lock()
        self._waiter_queue = PriorityQueue()

    def acquire(self, priority=0):
        self._mutex.acquire()
        # First, just check the lock.
        if self._is_available:
            self._is_available = False
            self._mutex.release()
            return True
        condition = Condition()
        condition.acquire()
        self._waiter_queue.put((priority, condition))
        self._mutex.release()
        condition.wait()
        condition.release()
        return True

    def release(self):
        self._mutex.acquire()
        # Notify the next thread in line, if any.
        try:
            _, condition = self._waiter_queue.get_nowait()
        except Empty:
            self._is_available = True
        else:
            condition.acquire()
            condition.notify()
            condition.release()
        self._mutex.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()


class Proc(object):
    _file_writer = None

    @property
    def _writer(self):
        if Proc._file_writer is None:
            Proc._file_writer = tf.summary.FileWriter(FLAGS.logdir)
        return Proc._file_writer

    def _build_map_summary(self, estimate_maps, space_map, goal_maps, reward_maps, value_maps, postfix=''):
        def _readout(image):
            image = image.astype(np.float32)
            image = image * 255
            image = image.astype(np.uint8)

            _, image = cv2.imencode('.png', image)
            return image.tostring()

        est_maps = [tf.Summary.Value(tag='losses/free_space_estimates_{}{}'.format(scale, postfix),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(estimate_maps)]
        opt_maps = [tf.Summary.Value(tag='losses/free_space_ground_truth{}'.format(postfix),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(space_map[-1]),
                                         height=space_map[-1].shape[0],
                                         width=space_map[-1].shape[1]))]
        gol_maps = [tf.Summary.Value(tag='losses/goal_{}{}'.format(scale, postfix),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(goal_maps)]
        fse_maps = [tf.Summary.Value(tag='losses/rewards_{}{}'.format(scale, postfix),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(reward_maps)]
        val_maps = [tf.Summary.Value(tag='losses/values_{}{}'.format(scale, postfix),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(value_maps)]

        return tf.Summary(value=est_maps + opt_maps + gol_maps + fse_maps + val_maps)

    def _build_trajectory_summary(self, rewards_history, info_history, exp, postfix=''):
        image = np.ones((28 + exp._width * 100, 28 + exp._height * 100, 3), dtype=np.uint8) * 255

        def _node_to_game_coordinate(node):
            row, col = node
            return 14 + int((col - 0.5) * 100), 14 + int((row - 0.5) * 100)

        def _pose_to_game_coordinate(pose):
            x, y = pose[:2]
            return 14 + int(x), 14 + image.shape[1] - int(y)

        if exp._env_name != info_history[-1]['env_name']:
            exp._build_free_space_estimate(info_history[-1]['env_name'])

        cv2.putText(image, exp._env_name, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        for row, col in exp._walls:
            loc = np.array([col, row])
            points = [loc, loc + np.array([0, 1]),
                      loc + np.array([1, 1]), loc + np.array([1, 0])]
            points = np.array([pts * 100 + np.array([14, 14]) for pts in points])
            cv2.fillConvexPoly(image, points, (224, 172, 52))

        for info in info_history:
            cv2.circle(image, _node_to_game_coordinate(info['GOAL.LOC']), 10, (82, 82, 255), -1)
            cv2.circle(image, _node_to_game_coordinate(info['SPAWN.LOC']), 10, (211, 111, 112), -1)
            cv2.circle(image, _pose_to_game_coordinate(info['POSE']), 4, (63, 121, 255), -1)

        encoded = cv2.imencode('.png', image)[1].tostring()

        return tf.Summary(value=[tf.Summary.Value(tag='losses/trajectory{}'.format(postfix),
                                                  image=tf.Summary.Image(encoded_image_string=encoded,
                                                                         height=image.shape[0],
                                                                         width=image.shape[1])),
                                 tf.Summary.Value(tag='losses/reward', simple_value=sum(rewards_history))])

    def _build_loss_summary(self, loss, postfix=''):
        return tf.Summary(value=[tf.Summary.Value(tag='losses/mean_loss{}'.format(postfix), simple_value=loss)])

    def _build_walltime_summary(self, begin, data, end):
        return tf.Summary(value=[tf.Summary.Value(tag='time/DAGGER_eval_walltime', simple_value=(data - begin)),
                                 tf.Summary.Value(tag='time/DAGGER_train_walltime', simple_value=(end - data)),
                                 tf.Summary.Value(tag='time/DAGGER_complete_walltime', simple_value=(end - begin))])

    def _build_gradient_summary(self, gradient_names, gradient_collections):
        gradient_means = np.array(gradient_collections).mean(axis=0).tolist()
        return tf.Summary(value=[tf.Summary.Value(tag='gradient/{}'.format(var), simple_value=val)
                                 for var, val in zip(gradient_names, gradient_means)])


class Worker(Proc):
    def __init__(self, saver, model, maps, global_step, eval=False):
        super(Worker, self).__init__()

        self._maps = maps
        self._saver = saver
        self._net = model

        tensors = model.intermediate_tensors

        self._estimate_maps = [tf.nn.sigmoid(estimate[0, -1, :, :, :4]) for estimate in tensors['estimate_map_list']]
        self._goal_maps = [tf.nn.sigmoid(goal[0, -1, :, :, :4]) for goal in tensors['goal_map_list']]
        self._reward_maps = [tf.nn.sigmoid(reward[0, -1, :, :, :FLAGS.vin_rewards]) for reward in
                             tensors['reward_map_list']]
        self._value_maps = [tf.nn.sigmoid(value[0, -1, :, :, :4]) for value in tensors['value_map_list']]
        self._action_maps = [tf.nn.sigmoid(action[0, -1, :, :, :4]) for action in tensors['action_map_list']]

        self._step_history = tf.placeholder(tf.string, name='step_history')
        self._step_history_op = tf.summary.text('game/step_history', self._step_history, collections=['game'])

        self._global_step = global_step
        self._update_global_step_op = tf.assign_add(global_step, 1)

        self._graph_lock = Lock()
        self._model_path = None

        self._eval = eval

    def _update_graph(self, sess, saver_lock):
        model_path = tf.train.latest_checkpoint(FLAGS.logdir)

        with self._graph_lock, saver_lock:
            if model_path is not None and model_path != self._model_path:
                self._saver.restore(sess, model_path)
                self._model_path = model_path

    def _merge_depth(self, obs, depth):
        return np.concatenate([obs, np.expand_dims(depth, axis=2)], axis=2) / 255.

    def __call__(self, lock, history, sess, coord):
        assert isinstance(history, dict)
        assert isinstance(sess, tf.Session)
        assert isinstance(coord, tf.train.Coordinator)

        history_lock, saver_lock = lock

        env = environment.get_game_environment(self._maps,
                                               multiproc=FLAGS.multiproc,
                                               random_goal=FLAGS.random_goal,
                                               random_spawn=FLAGS.random_spawn,
                                               apple_prob=FLAGS.apple_prob)
        exp = Expert()

        with coord.stop_on_exception():
            with sess.as_default(), sess.graph.as_default():
                while not coord.should_stop():
                    self._update_graph(sess, saver_lock)

                    np_global_step = sess.run(self._update_global_step_op)

                    random_rate = FLAGS.supervision_rate * (FLAGS.decay ** np_global_step)
                    if FLAGS.learn_mapper:
                        random_rate = 2
                    if FLAGS.eval:
                        random_rate = 0

                    env.reset()
                    obs, info = env.observations()

                    episode = dict()
                    episode['act'] = [np.argmax(exp.get_optimal_action(info))]
                    episode['obs'] = [self._merge_depth(obs, info['depth'])]
                    episode['ego'] = [[0., 0., 0.]]
                    episode['est'] = [exp.get_free_space_map(info, estimate_size=FLAGS.estimate_size)]
                    episode['gol'] = [exp.get_goal_map(info, estimate_size=FLAGS.estimate_size)]
                    episode['rwd'] = [0.]
                    episode['inf'] = [deepcopy(info)]

                    for _ in xrange(FLAGS.episode_size):
                        prev_info = deepcopy(episode['inf'][-1])
                        optimal_action = exp.get_optimal_action(prev_info)

                        if np.random.rand() < random_rate and not self._eval:
                            dagger_action = optimal_action
                        else:
                            expand_dim = lambda x: np.array([x])
                            feed_data = {'sequence_length': expand_dim(len(episode['obs'])),
                                         'visual_input': expand_dim(episode['obs']),
                                         'egomotion': expand_dim(episode['ego']),
                                         'reward': expand_dim(episode['rwd']),
                                         'space_map': expand_dim(episode['est']),
                                         'goal_map': expand_dim(episode['gol']),
                                         'estimate_map_list': [np.zeros(
                                             (1, FLAGS.estimate_size, FLAGS.estimate_size, 3))] * FLAGS.estimate_scale,
                                         'optimal_action': expand_dim(episode['act']),
                                         'optimal_estimate': expand_dim(episode['est']),
                                         'is_training': False}
                            feed_dict = prepare_feed_dict(self._net.input_tensors, feed_data)

                            results = sess.run(self._net.output_tensors['action'], feed_dict=feed_dict)

                            predict_action = np.squeeze(results)
                            dagger_action = predict_action

                        action = np.argmax(dagger_action)
                        obs, reward, terminal, info = env.step(action)

                        if not terminal:
                            episode['act'].append(np.argmax(optimal_action))
                            episode['obs'].append(self._merge_depth(obs, info['depth']))
                            episode['ego'].append(environment.calculate_egomotion(prev_info['POSE'], info['POSE']))
                            episode['est'].append(exp.get_free_space_map(info, estimate_size=FLAGS.estimate_size))
                            episode['gol'].append(exp.get_goal_map(info, estimate_size=FLAGS.estimate_size))
                            episode['rwd'].append(deepcopy(reward))
                            episode['inf'].append(deepcopy(info))
                        else:
                            break

                    if not self._eval:
                        with history_lock:
                            history_len = len(history['inf'])
                            history_idx = history_len % FLAGS.memory_size

                            for k, v in episode.iteritems():
                                if history_len < FLAGS.memory_size:
                                    history[k].append(v)
                                else:
                                    history[k][history_idx] = v

                    if np_global_step % FLAGS.save_every == 0 or self._eval:
                        expand_dim = lambda x: np.array([x])
                        feed_data = {'sequence_length': expand_dim(len(episode['obs'])),
                                     'visual_input': expand_dim(episode['obs']),
                                     'egomotion': expand_dim(episode['ego']),
                                     'reward': expand_dim(episode['rwd']),
                                     'space_map': expand_dim(episode['est']),
                                     'goal_map': expand_dim(episode['gol']),
                                     'estimate_map_list': [np.zeros(
                                         (1, FLAGS.estimate_size, FLAGS.estimate_size, 3))] * FLAGS.estimate_scale,
                                     'optimal_action': expand_dim(episode['act']),
                                     'optimal_estimate': expand_dim(episode['est']),
                                     'is_training': False}
                        feed_dict = prepare_feed_dict(self._net.input_tensors, feed_data)

                        summary_ops = self._estimate_maps + self._goal_maps + self._reward_maps + self._value_maps
                        results = sess.run(summary_ops, feed_dict=feed_dict)

                        estimate_maps_images = results[:len(self._estimate_maps)]
                        results = results[len(self._estimate_maps):]
                        goal_maps_images = results[:len(self._goal_maps)]
                        results = results[len(self._goal_maps):]
                        fused_maps_images = results[:len(self._reward_maps)]
                        results = results[len(self._reward_maps):]
                        value_maps_images = results[:len(self._value_maps)]
                        results = results[len(self._value_maps):]

                        assert len(results) == 0

                        postfix = '_eval' if self._eval else ''

                        self._writer.add_summary(self._build_map_summary(estimate_maps_images, episode['est'],
                                                                         goal_maps_images, fused_maps_images,
                                                                         value_maps_images, postfix),
                                                 global_step=np_global_step)

                        summary_text = ','.join('{}[{}]-{}={}'.format(key, idx, step, value)
                                                for step, info in enumerate(episode['inf'])
                                                for key in ('GOAL.LOC', 'SPAWN.LOC', 'POSE', 'env_name')
                                                for idx, value in enumerate(info[key]))
                        step_episode_summary = sess.run(self._step_history_op,
                                                        feed_dict={self._step_history: summary_text})
                        self._writer.add_summary(step_episode_summary, global_step=np_global_step)
                        self._writer.add_summary(self._build_trajectory_summary(episode['rwd'], episode['inf'], exp,
                                                                                postfix),
                                                 global_step=np_global_step)


class Trainer(Proc):
    def __init__(self, saver, model, global_step):
        super(Trainer, self).__init__()

        self._exp = Expert()
        self._net = model
        self._update_global_step_op = tf.assign_add(global_step, 1)
        self._enough_history = False

        optimizer_class = getattr(tf.train, FLAGS.optimizer)
        optimizer = optimizer_class(learning_rate=FLAGS.learning_rate)
        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        loss_key = 'loss' if not FLAGS.learn_mapper else 'estimate_loss'

        with tf.control_dependencies(self._update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(model.output_tensors[loss_key]))
            if FLAGS.grad_clip > 0:
                gradients_constrained, _ = tf.clip_by_global_norm(gradients, FLAGS.grad_clip)
            else:
                gradients_constrained = gradients
            self._gradient_names = [v.name for g, v in zip(gradients_constrained, variables) if g is not None]
            self._gradient_summary_op = [tf.reduce_mean(tf.abs(g)) for g in gradients_constrained if g is not None]
            self._train_op = optimizer.apply_gradients(zip(gradients_constrained, variables))

        with tf.control_dependencies([self._train_op]):
            self._train_loss = model.output_tensors[loss_key]

    def __call__(self, lock, history, sess, coord):
        assert isinstance(history, dict)
        assert isinstance(coord, tf.train.Coordinator)
        assert isinstance(self._writer, tf.summary.FileWriter)

        history_lock, saver_lock = lock

        with coord.stop_on_exception():
            with sess.as_default(), sess.graph.as_default():
                while not coord.should_stop():
                    np_global_step = sess.run(self._update_global_step_op)

                    if not self._enough_history:
                        with history_lock:
                            history_len = len(history['inf'])

                        while history_len < FLAGS.batch_size:
                            time.sleep(5)

                            with history_lock:
                                history_len = len(history['inf'])

                        self._enough_history = True

                    history_lock.acquire(-10)
                    batch_indices = random.sample(range(len(history['inf'])), FLAGS.batch_size)
                    batch_select = lambda x: [x[i] for i in batch_indices]

                    feed_data = {'sequence_length': batch_select([len(h) for h in history['inf']]),
                                 'visual_input': batch_select(history['obs']),
                                 'egomotion': batch_select(history['ego']),
                                 'reward': batch_select(history['rwd']),
                                 'space_map': batch_select(history['est']),
                                 'goal_map': batch_select(history['gol']),
                                 'estimate_map_list': [np.zeros((FLAGS.batch_size,
                                                                 FLAGS.estimate_size,
                                                                 FLAGS.estimate_size, 3))] * FLAGS.estimate_scale,
                                 'optimal_action': batch_select(history['act']),
                                 'optimal_estimate': batch_select(history['est']),
                                 'is_training': False}
                    history_lock.release()

                    feed_dict = prepare_feed_dict(self._net.input_tensors, feed_data)

                    gradient_collections = []
                    train_ops = [self._train_loss, self._train_op] + self._update_ops + self._gradient_summary_op
                    results = sess.run(train_ops, feed_dict=feed_dict)
                    loss = results[0]
                    gradient_collections.append(results[2 + len(self._update_ops):])

                    if np_global_step % FLAGS.save_every == 0:
                        self._writer.add_summary(self._build_loss_summary(loss), global_step=np_global_step)
                        self._writer.add_summary(self._build_gradient_summary(self._gradient_names,
                                                                              gradient_collections),
                                                 global_step=np_global_step)

                    if FLAGS.total_steps <= np_global_step:
                        coord.request_stop()


class ModelSaver(Proc):
    def __init__(self, saver, global_step):
        super(ModelSaver, self).__init__()

        self._saver = saver
        self._global_step = global_step
        self._last_save = None

    def __call__(self, lock, history, sess, coord):
        assert isinstance(history, dict)
        assert isinstance(coord, tf.train.Coordinator)
        assert isinstance(self._saver, tf.train.Saver)
        assert isinstance(self._writer, tf.summary.FileWriter)

        history_lock, saver_lock = lock

        with coord.stop_on_exception():
            with sess.as_default(), sess.graph.as_default():
                while not coord.should_stop():
                    np_global_step = sess.run(self._global_step)

                    if np_global_step % FLAGS.save_every == 0 and self._last_save != np_global_step:
                        checkpoint_path = '{}/step-{}.ckpt'.format(FLAGS.logdir, np_global_step)

                        with saver_lock:
                            self._saver.save(sess, checkpoint_path)

                            if tf.train.latest_checkpoint(FLAGS.logdir):
                                tf.train.update_checkpoint_state(FLAGS.logdir, checkpoint_path)
                            else:
                                tf.train.generate_checkpoint_state_proto(FLAGS.logdir, checkpoint_path)

                        self._last_save = np_global_step
                    else:
                        time.sleep(10)


def prepare_feed_dict(tensors, data):
    feed_dict = {}
    for k, v in tensors.iteritems():
        if k not in data:
            continue

        if not isinstance(v, list):
            if isinstance(data[k], np.ndarray):
                feed_dict[v] = data[k].astype(v.dtype.as_numpy_dtype)
            else:
                feed_dict[v] = data[k]
        else:
            for t, d in zip(v, data[k]):
                feed_dict[t] = d.astype(t.dtype.as_numpy_dtype)

    return feed_dict


def main(_):
    maps = FLAGS.maps.split(',')

    if FLAGS.worker_size > 0:
        maps_chunk = [','.join(maps[i::FLAGS.worker_size]) for i in xrange(FLAGS.worker_size)]

    params = vars(FLAGS)

    model_path = tf.train.latest_checkpoint(FLAGS.logdir)

    procs = []

    config = tf.ConfigProto(device_count={'GPU': 0})
    worker_sess = tf.Session(config=config, graph=tf.Graph())

    with worker_sess.as_default(), worker_sess.graph.as_default():
        explore_global_step = tf.get_variable('explore_global_step', shape=(), dtype=tf.int32,
                                              initializer=tf.constant_initializer(-1), trainable=False)

        worker_model = CMAP(**params)
        worker_saver = tf.train.Saver(var_list=slim.get_variables(scope='CMAP'))

        if FLAGS.worker_size > 0:
            for chunk in maps_chunk:
                procs.append((Worker(worker_saver, worker_model, chunk, explore_global_step), worker_sess))

        if FLAGS.eval:
            procs.append((Worker(worker_saver, worker_model, FLAGS.eval_maps, explore_global_step, True), worker_sess))

        worker_sess.run(tf.global_variables_initializer())

        if model_path is not None:
            worker_saver.restore(worker_sess, model_path)

    if FLAGS.worker_size > 0:
        trainer_sess = tf.Session(graph=tf.Graph())

        with trainer_sess.as_default(), trainer_sess.graph.as_default():
            train_global_step = tf.get_variable('train_global_step', shape=(), dtype=tf.int32,
                                                initializer=tf.constant_initializer(-1), trainable=False)
            trainer_model = CMAP(**params)
            trainer_saver = tf.train.Saver()

            procs.append((Trainer(trainer_saver, trainer_model, train_global_step), trainer_sess))
            procs.append((ModelSaver(trainer_saver, train_global_step), trainer_sess))

            trainer_sess.run(tf.global_variables_initializer())

            if model_path is not None:
                trainer_saver.restore(trainer_sess, model_path)

    history = dict()
    history['act'] = []
    history['obs'] = []
    history['ego'] = []
    history['est'] = []
    history['gol'] = []
    history['rwd'] = []
    history['inf'] = []

    history_lock = PriorityLock()
    saver_lock = PriorityLock()
    locks = (history_lock, saver_lock)

    try:
        coord = tf.train.Coordinator()
        threads = [Thread(target=proc, args=(locks, history, sess, coord)) for proc, sess in procs]
        for t in threads:
            assert isinstance(t, Thread)
            t.start()
        coord.join(threads)
    except Exception as e:
        coord.request_stop()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CMAP training')


    def DEFINE_arg(name, default, type):
        parser.add_argument('--{}'.format(name), default=default, type=type)


    DEFINE_string = lambda n, d, _: DEFINE_arg(n, d, str)
    DEFINE_boolean = lambda n, d, _: DEFINE_arg(n, d, bool)
    DEFINE_integer = lambda n, d, _: DEFINE_arg(n, d, int)
    DEFINE_float = lambda n, d, _: DEFINE_arg(n, d, float)

    DEFINE_string('maps', 'training-09x09-0001,training-09x09-0004,training-09x09-0005,training-09x09-0006,'
                          'training-09x09-0007,training-09x09-0008,training-09x09-0009,training-09x09-0010',
                  'Comma separated game environment list')
    DEFINE_string('eval_maps', 'training-09x09-0001,training-09x09-0004,training-09x09-0005,training-09x09-0006,'
                               'training-09x09-0007,training-09x09-0008,training-09x09-0009,training-09x09-0010',
                  'Comma separated game environment list')
    DEFINE_string('optimizer', 'RMSPropOptimizer', 'Tensorflow optimizer class')
    DEFINE_string('logdir', './output/dummy', 'Log directory')
    DEFINE_boolean('learn_mapper', False, 'Mapper supervised training')
    DEFINE_boolean('eval', True, 'Run evaluation')
    DEFINE_boolean('multiproc', True, 'Multiproc environment')
    DEFINE_boolean('random_goal', True, 'Allow random goal')
    DEFINE_boolean('random_spawn', True, 'Allow random spawn')
    DEFINE_integer('total_steps', 10 ** 8, 'Total number of training steps')
    DEFINE_integer('save_every', 5, 'Save every n steps')
    DEFINE_integer('memory_size', 10 ** 4, 'Max steps per episode')
    DEFINE_integer('episode_size', 10 ** 3, 'Max steps per episode')
    DEFINE_integer('batch_size', 1, 'Number of environments to run')
    DEFINE_integer('worker_size', 1, 'Number of workers')
    DEFINE_integer('evaluator_size', 1, 'Number of eval threads')
    DEFINE_float('apple_prob', 0.9, 'Apple probability')
    DEFINE_float('learning_rate', 0.001, 'ADAM learning rate')
    DEFINE_float('supervision_rate', 1., 'DAGGER supervision rate')
    DEFINE_float('decay', 0.99, 'DAGGER decay')
    DEFINE_float('grad_clip', 0, 'Gradient clipping value')

    for k, v in CMAP.params().iteritems():
        DEFINE_arg(k, v, type(v))

    FLAGS = parser.parse_args()

    if FLAGS.learn_mapper and FLAGS.eval:
        raise ValueError('bad configuration -- evaluate on mapper training?')

    tf.app.run()

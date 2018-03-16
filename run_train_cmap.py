import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
from expert import Expert
from model import CMAP
from copy import deepcopy
import time
import cv2

flags = tf.app.flags
flags.DEFINE_string('maps', 'training-09x09-0001,training-09x09-0004,training-09x09-0005,training-09x09-0006,'
                            'training-09x09-0007,training-09x09-0008,training-09x09-0009,training-09x09-0010',
                    'Comma separated game environment list')
flags.DEFINE_string('logdir', './output/dummy', 'Log directory')
flags.DEFINE_boolean('learn_mapper', False, 'Mapper supervised training')
flags.DEFINE_boolean('eval', False, 'Run evaluation')
flags.DEFINE_boolean('debug', False, 'Save debugging information')
flags.DEFINE_boolean('multiproc', True, 'Multiproc environment')
flags.DEFINE_boolean('random_goal', True, 'Allow random goal')
flags.DEFINE_boolean('random_spawn', True, 'Allow random spawn')
flags.DEFINE_integer('num_games', 10 ** 8, 'Number of games to play')
flags.DEFINE_integer('episode_size', 10 ** 3, 'Max steps per episode')
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run')
flags.DEFINE_float('apple_prob', 0.9, 'Apple probability')
flags.DEFINE_float('learning_rate', 0.001, 'ADAM learning rate')
flags.DEFINE_float('supervision_rate', 1., 'DAGGER supervision rate')
flags.DEFINE_float('decay', 0.99, 'DAGGER decay')
flags.DEFINE_float('grad_clip', 0, 'Gradient clipping value')
FLAGS = None


def DAGGER_train_step(sess, train_op, global_step, train_step_kwargs):
    game = train_step_kwargs['game']
    exp = train_step_kwargs['exp']
    net = train_step_kwargs['net']

    assert isinstance(exp, Expert)
    assert isinstance(net, CMAP)

    summary_writer = train_step_kwargs['summary_writer']
    update_ops = train_step_kwargs['update_ops']
    train_loss = train_step_kwargs['train_loss']
    step_history = train_step_kwargs['step_history']
    step_history_op = train_step_kwargs['step_history_op']
    gradient_names = train_step_kwargs['gradient_names']
    gradient_summary_op = train_step_kwargs['gradient_summary_op']
    update_global_step_op = train_step_kwargs['update_global_step_op']
    estimate_maps = train_step_kwargs['estimate_maps']
    goal_maps = train_step_kwargs['goal_maps']
    reward_maps = train_step_kwargs['reward_maps']
    value_maps = train_step_kwargs['value_maps']

    def _build_map_summary(estimate_maps, space_map, goal_maps, reward_maps, value_maps):
        def _readout(image):
            image = image.astype(np.float32)
            image = image * 255
            image = image.astype(np.uint8)

            _, image = cv2.imencode('.png', image)
            return image.tostring()

        est_maps = [tf.Summary.Value(tag='losses/free_space_estimates_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(estimate_maps)]
        opt_maps = [tf.Summary.Value(tag='losses/free_space_ground_truth',
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(space_map[-1]),
                                         height=space_map[-1].shape[0],
                                         width=space_map[-1].shape[1]))]
        gol_maps = [tf.Summary.Value(tag='losses/goal_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(goal_maps)]
        fse_maps = [tf.Summary.Value(tag='losses/rewards_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(reward_maps)]
        val_maps = [tf.Summary.Value(tag='losses/values_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=_readout(image),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(value_maps)]

        return tf.Summary(value=est_maps + opt_maps + gol_maps + fse_maps + val_maps)

    def _build_trajectory_summary(rate, loss, rewards_history, info_history, exp):
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

        return tf.Summary(value=[tf.Summary.Value(tag='losses/trajectory',
                                                  image=tf.Summary.Image(encoded_image_string=encoded,
                                                                         height=image.shape[0],
                                                                         width=image.shape[1])),
                                 tf.Summary.Value(tag='losses/supervision_rate', simple_value=rate),
                                 tf.Summary.Value(tag='losses/average_loss_per_step', simple_value=loss),
                                 tf.Summary.Value(tag='losses/reward', simple_value=sum(rewards_history))])

    def _build_walltime_summary(begin, data, end):
        return tf.Summary(value=[tf.Summary.Value(tag='time/DAGGER_eval_walltime', simple_value=(data - begin)),
                                 tf.Summary.Value(tag='time/DAGGER_train_walltime', simple_value=(end - data)),
                                 tf.Summary.Value(tag='time/DAGGER_complete_walltime', simple_value=(end - begin))])

    def _build_gradient_summary(gradient_names, gradient_collections):
        gradient_means = np.array(gradient_collections).mean(axis=0).tolist()
        return tf.Summary(value=[tf.Summary.Value(tag='gradient/{}'.format(var), simple_value=val)
                                 for var, val in zip(gradient_names, gradient_means)])

    def _merge_depth(obs, depth):
        return np.concatenate([obs, np.expand_dims(depth, axis=2)], axis=2) / 255.

    train_step_start = time.time()

    np_global_step = sess.run(global_step)

    random_rate = FLAGS.supervision_rate * (FLAGS.decay ** np_global_step)
    if FLAGS.learn_mapper:
        random_rate = 2
    if FLAGS.eval:
        random_rate = 0

    history = {}
    for his_type in 'act obs ego est gol rwd inf'.split():
        history[his_type] = []

    # Dataset aggregation
    for batch_index, env in enumerate(game):
        env.reset()
        obs, info = env.observations()

        history['act'].append([np.argmax(exp.get_optimal_action(info))])
        history['obs'].append([_merge_depth(obs, info['depth'])])
        history['ego'].append([[0., 0., 0.]])
        history['est'].append([exp.get_free_space_map(info)])
        history['gol'].append([exp.get_goal_map(info)])
        history['rwd'].append([0.])
        history['inf'].append([deepcopy(info)])

        for _ in xrange(FLAGS.episode_size):
            prev_info = deepcopy(history['inf'][batch_index][-1])
            optimal_action = exp.get_optimal_action(prev_info)

            if np.random.rand() < random_rate:
                dagger_action = optimal_action
            else:
                expand_dim = lambda x: np.array([x])
                feed_data = {'sequence_length': expand_dim(len(history['obs'][batch_index])),
                             'visual_input': expand_dim(history['obs'][batch_index]),
                             'egomotion': expand_dim(history['ego'][batch_index]),
                             'reward': expand_dim(history['rwd'][batch_index]),
                             'space_map': expand_dim(history['est'][batch_index]),
                             'goal_map': expand_dim(history['gol'][batch_index]),
                             'estimate_map_list': [np.zeros((1, 256, 256, 3))] * net._estimate_scale,
                             'optimal_action': expand_dim(history['act'][batch_index]),
                             'optimal_estimate': expand_dim(history['est'][batch_index]),
                             'is_training': False}
                feed_dict = prepare_feed_dict(net.input_tensors, feed_data)

                results = sess.run(net.output_tensors['action'], feed_dict=feed_dict)

                predict_action = np.squeeze(results)
                dagger_action = predict_action

            action = np.argmax(dagger_action)
            obs, reward, terminal, info = env.step(action)

            if not terminal:
                history['act'][batch_index].append(np.argmax(optimal_action))
                history['obs'][batch_index].append(_merge_depth(obs, info['depth']))
                history['ego'][batch_index].append(environment.calculate_egomotion(prev_info['POSE'], info['POSE']))
                history['est'][batch_index].append(exp.get_free_space_map(info))
                history['gol'][batch_index].append(exp.get_goal_map(info))
                history['rwd'][batch_index].append(deepcopy(reward))
                history['inf'][batch_index].append(deepcopy(info))
            else:
                break

    train_step_eval = time.time()

    for v in history.itervalues():
        assert len(v) == FLAGS.batch_size

    # Summary
    feed_data = {'sequence_length': np.array([len(h) for h in history['inf']]),
                 'visual_input': np.array(history['obs']),
                 'egomotion': np.array(history['ego']),
                 'reward': np.array(history['rwd']),
                 'space_map': np.array(history['est']),
                 'goal_map': np.array(history['gol']),
                 'estimate_map_list': [np.zeros((FLAGS.batch_size, 256, 256, 3))] * net._estimate_scale,
                 'optimal_action': np.array(history['act']),
                 'optimal_estimate': np.array(history['est']),
                 'is_training': False}
    feed_dict = prepare_feed_dict(net.input_tensors, feed_data)

    results = sess.run([train_loss] + estimate_maps + goal_maps + reward_maps + value_maps, feed_dict=feed_dict)

    cumulative_loss = results[0]
    results = results[1:]
    estimate_maps_images = results[:len(estimate_maps)]
    results = results[len(estimate_maps):]
    goal_maps_images = results[:len(goal_maps)]
    results = results[len(goal_maps):]
    fused_maps_images = results[:len(reward_maps)]
    results = results[len(reward_maps):]
    value_maps_images = results[:len(value_maps)]
    results = results[len(value_maps):]

    assert len(results) == 0

    # Training
    if not FLAGS.eval:
        gradient_collections = []
        train_ops = [train_loss, train_op] + update_ops + gradient_summary_op
        results = sess.run(train_ops, feed_dict=feed_dict)
        cumulative_loss = results[0]
        gradient_collections.append(results[2 + len(update_ops):])

    train_step_end = time.time()

    summary_text = ','.join('{}[{}]-{}={}'.format(key, idx, step, value)
                            for step, info in enumerate(history['inf'][0])
                            for key in ('GOAL.LOC', 'SPAWN.LOC', 'POSE', 'env_name')
                            for idx, value in enumerate(info[key]))
    step_history_summary, new_global_step = sess.run([step_history_op, update_global_step_op],
                                                     feed_dict={step_history: summary_text})
    summary_writer.add_summary(step_history_summary, global_step=np_global_step)

    summary_writer.add_summary(_build_map_summary(estimate_maps_images, history['est'][0], goal_maps_images,
                                                  fused_maps_images, value_maps_images),
                               global_step=np_global_step)

    if not FLAGS.eval:
        summary_writer.add_summary(_build_gradient_summary(gradient_names, gradient_collections),
                                   global_step=np_global_step)

    summary_writer.add_summary(_build_trajectory_summary(random_rate, cumulative_loss,
                                                         history['rwd'][0], history['inf'][0], exp),
                               global_step=np_global_step)
    summary_writer.add_summary(_build_walltime_summary(train_step_start, train_step_eval, train_step_end),
                               global_step=np_global_step)

    should_stop = new_global_step >= FLAGS.num_games

    return cumulative_loss, should_stop


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
    tf.reset_default_graph()

    maps = FLAGS.maps.split(',')
    game = [environment.get_game_environment(','.join(maps[i::FLAGS.batch_size]),
                                             multiproc=FLAGS.multiproc,
                                             random_goal=FLAGS.random_goal,
                                             random_spawn=FLAGS.random_spawn,
                                             apple_prob=FLAGS.apple_prob)
            for i in xrange(FLAGS.batch_size)]
    net = CMAP(**FLAGS.__flags)

    estimate_images = [tf.nn.sigmoid(estimate[0, -1, :, :, :4]) for estimate in
                       net.intermediate_tensors['estimate_map_list']]
    goal_images = [tf.nn.sigmoid(goal[0, -1, :, :, :4]) for goal in net.intermediate_tensors['goal_map_list']]
    reward_images = [tf.nn.sigmoid(reward[0, -1, :, :, :FLAGS.vin_rewards])
                     for reward in net.intermediate_tensors['reward_map_list']]
    value_images = [tf.nn.sigmoid(value[0, -1, :, :, :4]) for value in net.intermediate_tensors['value_map_list']]
    action_images = [tf.nn.sigmoid(action[0, -1, :, :, :4]) for action in net.intermediate_tensors['action_map_list']]

    step_history = tf.placeholder(tf.string, name='step_history')
    step_history_op = tf.summary.text('game/step_history', step_history, collections=['game'])

    global_step = slim.get_or_create_global_step()
    update_global_step_op = tf.assign_add(global_step, 1)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    loss_key = 'loss' if not FLAGS.learn_mapper else 'estimate_loss'

    with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(net.output_tensors[loss_key]))
        if FLAGS.grad_clip > 0:
            gradients_constrained, _ = tf.clip_by_global_norm(gradients, FLAGS.grad_clip)
        else:
            gradients_constrained = gradients
        gradient_names = [v.name for g, v in zip(gradients_constrained, variables) if g is not None]
        gradient_summary_op = [tf.reduce_mean(tf.abs(g)) for g in gradients_constrained if g is not None]
        train_op = optimizer.apply_gradients(zip(gradients_constrained, variables))

    with tf.control_dependencies([train_op]):
        train_loss = net.output_tensors[loss_key]

    if FLAGS.eval:
        model_path = tf.train.latest_checkpoint(FLAGS.logdir)
        init_op = tf.variables_initializer([global_step])
        load_op, load_feed_dict = slim.assign_from_checkpoint(model_path,
                                                              slim.get_variables_to_restore(exclude=[global_step.name]))

        init_op = tf.group(init_op, load_op)

    slim.learning.train(train_op=train_op,
                        logdir=FLAGS.logdir if not FLAGS.eval else '{}_eval'.format(model_path),
                        init_op=init_op if FLAGS.eval else slim.learning._USE_DEFAULT,
                        init_feed_dict=load_feed_dict if FLAGS.eval else None,
                        global_step=global_step,
                        train_step_fn=DAGGER_train_step,
                        train_step_kwargs=dict(game=game, exp=Expert(), net=net,
                                               update_ops=update_ops,
                                               train_loss=train_loss,
                                               update_global_step_op=update_global_step_op,
                                               step_history=step_history,
                                               step_history_op=step_history_op,
                                               gradient_names=gradient_names,
                                               gradient_summary_op=gradient_summary_op,
                                               estimate_maps=estimate_images,
                                               goal_maps=goal_images,
                                               reward_maps=reward_images,
                                               value_maps=value_images,
                                               action_maps=action_images),
                        number_of_steps=FLAGS.num_games,
                        save_interval_secs=300 if not FLAGS.debug else 60,
                        save_summaries_secs=300 if not FLAGS.debug else 60)


if __name__ == '__main__':
    for k, v in CMAP.params().iteritems():
        if isinstance(v, bool):
            flags.DEFINE_boolean(k, v, 'CMAP bool parameter')
        elif isinstance(v, int):
            flags.DEFINE_integer(k, v, 'CMAP int parameter')
        elif isinstance(v, float):
            flags.DEFINE_float(k, v, 'CMAP float parameter')
        elif isinstance(v, str):
            flags.DEFINE_string(k, v, 'CMAP str parameter')

    FLAGS = flags.FLAGS

    if FLAGS.learn_mapper and FLAGS.eval:
        raise ValueError('bad configuration -- evaluate on mapper training?')

    tf.app.run()

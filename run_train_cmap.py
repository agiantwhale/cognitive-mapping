import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
import expert
from model import CMAP
import os
import copy
import time
import cv2

flags = tf.app.flags
flags.DEFINE_string('maps', 'training-09x09-0127', 'Comma separated game environment list')
flags.DEFINE_string('logdir', './output/dummy', 'Log directory')
flags.DEFINE_boolean('debug', False, 'Save debugging information')
flags.DEFINE_boolean('multiproc', False, 'Multiproc environment')
flags.DEFINE_boolean('random_goal', True, 'Allow random goal')
flags.DEFINE_boolean('random_spawn', True, 'Allow random spawn')
flags.DEFINE_integer('max_steps_per_episode', 10 ** 100, 'Max steps per episode')
flags.DEFINE_integer('num_games', 10 ** 8, 'Number of games to play')
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run')
flags.DEFINE_integer('history_size', 32, 'Number of environments to run')
flags.DEFINE_integer('estimate_scale', 3, 'Number of hierarchies')
flags.DEFINE_integer('vin_iterations', 10, 'Number of VIN iterations to run')
flags.DEFINE_float('apple_prob', 0.9, 'Apple probability')
flags.DEFINE_float('learning_rate', 0.001, 'ADAM learning rate')
flags.DEFINE_float('supervision_rate', 1., 'DAGGER supervision rate')
flags.DEFINE_float('decay', 0.99, 'DAGGER decay')
flags.DEFINE_float('grad_clip', 1., 'Gradient clipping value')
FLAGS = flags.FLAGS


def DAGGER_train_step(sess, train_op, global_step, train_step_kwargs):
    env = train_step_kwargs['env']
    exp = train_step_kwargs['exp']
    net = train_step_kwargs['net']
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

    def _build_map_summary(estimate_maps, goal_maps, reward_maps, value_maps):
        def _readout(image):
            image += 0.001
            image = np.exp(image / np.max(image))
            image = np.abs((image - np.min(image)) / (1 + np.max(image) - np.min(image))) * 255
            image = image.astype(np.uint8)
            return image

        est_maps = [tf.Summary.Value(tag='losses/free_space_estimates_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=cv2.imencode('.png', _readout(image))[1].tostring(),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(estimate_maps[-1])]
        gol_maps = [tf.Summary.Value(tag='losses/goal_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=cv2.imencode('.png', _readout(image))[1].tostring(),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(goal_maps[-1])]
        fse_maps = [tf.Summary.Value(tag='losses/rewards_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=cv2.imencode('.png', _readout(image))[1].tostring(),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(reward_maps[-1])]
        val_maps = [tf.Summary.Value(tag='losses/values_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=cv2.imencode('.png', _readout(image))[1].tostring(),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, image in enumerate(value_maps[-1])]

        return tf.Summary(value=est_maps + gol_maps + fse_maps + val_maps)

    def _build_trajectory_summary(rate, loss, rewards_history, info_history, exp):
        image = np.ones((28 + exp._width * 100, 28 + exp._height * 100, 3), dtype=np.uint8) * 255

        def _node_to_game_coordinate(node):
            row, col = node
            return 14 + int((col - 0.5) * 100), 14 + int((row - 0.5) * 100)

        def _pose_to_game_coordinate(pose):
            x, y = pose[:2]
            return 14 + int(x), 14 + image.shape[1] - int(y)

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

    env.reset()
    obs, info = env.observations()

    optimal_action_history = [np.argmax(exp.get_optimal_action(info))]
    observation_history = [_merge_depth(obs, info['depth'])]
    egomotion_history = [[0., 0., 0.]]
    goal_map_history = [exp.get_goal_map(info)]
    rewards_history = [0.]
    estimate_maps_history = [[np.zeros((1, 64, 64, 3))] * net._estimate_scale]
    info_history = [info]

    estimate_maps_images = []
    goal_maps_images = []
    fused_maps_images = []
    value_maps_images = []

    # Dataset aggregation
    terminal = False
    while not terminal and len(info_history) < FLAGS.max_steps_per_episode:
        _, previous_info = env.observations()
        previous_info = copy.deepcopy(previous_info)

        feed_dict = prepare_feed_dict(net.input_tensors, {'sequence_length': np.array([1]),
                                                          'visual_input': np.array([[observation_history[-1]]]),
                                                          'egomotion': np.array([[egomotion_history[-1]]]),
                                                          'reward': np.array([[rewards_history[-1]]]),
                                                          'goal_map': np.array([[goal_map_history[-1]]]),
                                                          'estimate_map_list': estimate_maps_history[-1],
                                                          'is_training': False})

        results = sess.run([net.output_tensors['action']] +
                           estimate_maps +
                           goal_maps +
                           reward_maps +
                           value_maps +
                           net.intermediate_tensors['estimate_map_list'], feed_dict=feed_dict)
        predict_action = np.squeeze(results[0])
        optimal_action = exp.get_optimal_action(previous_info)
        dagger_action = optimal_action if np.random.rand() < random_rate else predict_action

        action = np.argmax(dagger_action)
        obs, reward, terminal, info = env.step(action)

        maps_count = len(estimate_maps) + len(goal_maps) + len(reward_maps) + len(value_maps)

        optimal_action_history.append(np.argmax(optimal_action))
        observation_history.append(_merge_depth(obs, info['depth']))
        egomotion_history.append(environment.calculate_egomotion(previous_info['POSE'], info['POSE']))
        goal_map_history.append(exp.get_goal_map(info))
        rewards_history.append(copy.deepcopy(reward))
        estimate_maps_history.append([tensor[:, 0, :, :, :] for tensor in results[1 + maps_count:]])
        info_history.append(copy.deepcopy(info))

        idx = 1
        estimate_maps_images.append(results[idx:idx + len(estimate_maps)])
        idx += len(estimate_maps)
        goal_maps_images.append(results[idx:idx + len(goal_maps)])
        idx += len(goal_maps)
        fused_maps_images.append(results[idx:idx + len(reward_maps)])
        idx += len(reward_maps)
        value_maps_images.append(results[idx:idx + len(value_maps)])
        idx += len(value_maps)

        assert idx == (maps_count + 1)

    train_step_eval = time.time()

    assert len(optimal_action_history) == len(observation_history) == len(egomotion_history) == len(rewards_history)

    # Training
    indices = np.arange(FLAGS.history_size, len(optimal_action_history) + 1)
    np.random.shuffle(indices)

    gradient_collections = []
    cumulative_loss = 0
    for i in xrange(0, len(info_history), FLAGS.batch_size):
        batch_indices = indices[i:i + FLAGS.batch_size].tolist()
        batch_size = len(batch_indices)

        if batch_size == 0:
            break

        sequence_length = np.array([FLAGS.history_size] * batch_size)
        concat_observation_history = [observation_history[ind - FLAGS.history_size:ind] for ind in batch_indices]
        concat_egomotion_history = [egomotion_history[ind - FLAGS.history_size:ind] for ind in batch_indices]
        concat_goal_map_history = [goal_map_history[ind - FLAGS.history_size:ind] for ind in batch_indices]
        concat_reward_history = [rewards_history[ind - FLAGS.history_size:ind] for ind in batch_indices]
        concat_optimal_action_history = [optimal_action_history[ind - FLAGS.history_size:ind] for ind in batch_indices]
        # concat_estimate_map_list = [[] for _ in xrange(net._estimate_scale)]
        # for ind in batch_indices:
        #     for idx, estimate_map in enumerate(estimate_maps_history[ind - FLAGS.history_size]):
        #         concat_estimate_map_list[idx].append(estimate_map)
        # concat_estimate_map_list = [np.concatenate(map_list, axis=0) for map_list in concat_estimate_map_list]
        concat_estimate_map_list = [np.zeros((batch_size, 64, 64, 3)) for _ in xrange(net._estimate_scale)]

        feed_dict = prepare_feed_dict(net.input_tensors, {'sequence_length': sequence_length,
                                                          'visual_input': np.array(concat_observation_history),
                                                          'egomotion': np.array(concat_egomotion_history),
                                                          'reward': np.array(concat_reward_history),
                                                          'optimal_action': np.array(concat_optimal_action_history),
                                                          'goal_map': np.stack(concat_goal_map_history, axis=0),
                                                          'estimate_map_list': concat_estimate_map_list,
                                                          'is_training': True})

        train_ops = [train_loss, train_op] + update_ops + gradient_summary_op

        results = sess.run(train_ops, feed_dict=feed_dict)
        cumulative_loss += results[0]
        gradient_collections.append(results[2 + len(update_ops):])

    cumulative_loss /= len(optimal_action_history)

    train_step_end = time.time()

    summary_text = ','.join('{}[{}]-{}={}'.format(key, idx, step, value)
                            for step, info in enumerate(info_history)
                            for key in ('GOAL.LOC', 'SPAWN.LOC', 'POSE', 'env_name')
                            for idx, value in enumerate(info[key]))
    step_history_summary, new_global_step = sess.run([step_history_op, update_global_step_op],
                                                     feed_dict={step_history: summary_text})
    summary_writer.add_summary(step_history_summary, global_step=np_global_step)

    summary_writer.add_summary(_build_map_summary(estimate_maps_images, goal_maps_images,
                                                  fused_maps_images, value_maps_images),
                               global_step=np_global_step)
    summary_writer.add_summary(_build_gradient_summary(gradient_names, gradient_collections),
                               global_step=np_global_step)
    summary_writer.add_summary(_build_trajectory_summary(random_rate, cumulative_loss,
                                                         rewards_history, info_history, exp),
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

    env = environment.get_game_environment(FLAGS.maps,
                                           multiproc=FLAGS.multiproc,
                                           random_goal=FLAGS.random_goal,
                                           random_spawn=FLAGS.random_spawn,
                                           apple_prob=FLAGS.apple_prob)
    exp = expert.Expert()
    net = CMAP(num_iterations=FLAGS.vin_iterations,
               estimate_scale=FLAGS.estimate_scale)

    estimate_images = [estimate[0, -1, :, :, 0] for estimate in net.intermediate_tensors['estimate_map_list']]
    goal_images = [goal[0, -1, :, :, 0] for goal in net.intermediate_tensors['goal_map_list']]
    reward_images = [reward[0, -1, :, :, 0] for reward in net.intermediate_tensors['reward_map_list']]
    value_images = [value[0, -1, :, :, 0] for value in net.intermediate_tensors['value_map_list']]
    action_images = [action[0, -1, :, :, 0] for action in net.intermediate_tensors['action_map_list']]

    step_history = tf.placeholder(tf.string, name='step_history')
    step_history_op = tf.summary.text('game/step_history', step_history, collections=['game'])

    global_step = slim.get_or_create_global_step()
    update_global_step_op = tf.assign_add(global_step, 1)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        gradients = optimizer.compute_gradients(net.output_tensors['loss'])
        gradients_constrained = [(tf.clip_by_value(g, -1 * FLAGS.grad_clip, FLAGS.grad_clip), v) for g, v in gradients]
        gradient_names = [v.name for _, v in gradients]
        gradient_summary_op = [tf.reduce_mean(tf.abs(g)) for g, _ in gradients_constrained]
        train_op = optimizer.apply_gradients(gradients_constrained)

    with tf.control_dependencies([train_op]):
        train_loss = net.output_tensors['loss']

    slim.learning.train(train_op=train_op,
                        logdir=FLAGS.logdir,
                        global_step=global_step,
                        train_step_fn=DAGGER_train_step,
                        train_step_kwargs=dict(env=env, exp=exp, net=net,
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
    tf.app.run()

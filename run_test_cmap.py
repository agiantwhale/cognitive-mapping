import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
import expert
from model import CMAP
import copy
import cv2

flags = tf.app.flags
flags.DEFINE_string('maps', 'training-09x09-0127', 'Comma separated game environment list')
flags.DEFINE_string('modeldir', './output/dummy', 'Model directory')
flags.DEFINE_string('logdir', './output/dummy_test', 'Log directory')
flags.DEFINE_boolean('debug', False, 'Save debugging information')
flags.DEFINE_boolean('multiproc', False, 'Multiproc environment')
flags.DEFINE_boolean('random_goal', True, 'Allow random goal')
flags.DEFINE_boolean('random_spawn', True, 'Allow random spawn')
flags.DEFINE_integer('max_steps_per_episode', 10 ** 100, 'Max steps per episode')
flags.DEFINE_integer('num_games', 10 ** 8, 'Number of games to play')
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run')
FLAGS = flags.FLAGS


def DAGGER_train_step(sess, train_op, global_step, train_step_kwargs):
    env = train_step_kwargs['env']
    exp = train_step_kwargs['exp']
    net = train_step_kwargs['net']
    summary_writer = train_step_kwargs['summary_writer']
    step_history = train_step_kwargs['step_history']
    step_history_op = train_step_kwargs['step_history_op']
    update_global_step_op = train_step_kwargs['update_global_step_op']
    estimate_maps = train_step_kwargs['estimate_maps']
    value_maps = train_step_kwargs['value_maps']

    def _build_map_summary(estimate_maps, value_maps):
        def _to_image(img):
            return (np.expand_dims(np.squeeze(img), axis=2) * 255).astype(np.uint8)

        est_maps = [tf.Summary.Value(tag='losses/free_space_estimates_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=cv2.imencode('.png', image)[1].tostring(),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, map in enumerate(estimate_maps[-1])
                    for image in (_to_image(map),)]
        val_maps = [tf.Summary.Value(tag='losses/values_{}'.format(scale),
                                     image=tf.Summary.Image(
                                         encoded_image_string=cv2.imencode('.png', image)[1].tostring(),
                                         height=image.shape[0],
                                         width=image.shape[1]))
                    for scale, map in enumerate(value_maps[-1])
                    for image in (_to_image(map),)]

        return tf.Summary(value=est_maps + val_maps)

    def _build_trajectory_summary(loss, rewards_history, info_history, exp):
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

        cv2.imshow('trajectory', image)
        cv2.waitKey(-1)

        encoded = cv2.imencode('.png', image)[1].tostring()

        return tf.Summary(value=[tf.Summary.Value(tag='losses/trajectory',
                                                  image=tf.Summary.Image(encoded_image_string=encoded,
                                                                         height=image.shape[0],
                                                                         width=image.shape[1])),
                                 tf.Summary.Value(tag='losses/average_loss_per_step', simple_value=loss),
                                 tf.Summary.Value(tag='losses/reward', simple_value=sum(rewards_history))])

    def _merge_depth(obs, depth):
        return np.concatenate([obs, np.expand_dims(depth, axis=2)], axis=2)

    np_global_step = sess.run(global_step)

    env.reset()
    obs, info = env.observations()

    cumulative_loss = 0
    optimal_action_history = [exp.get_optimal_action(info)]
    observation_history = [_merge_depth(obs, info['depth'])]
    egomotion_history = [[0., 0.]]
    rewards_history = [0.]
    estimate_maps_history = [[np.zeros((1, 64, 64, 3))] * net._estimate_scale]
    info_history = [info]

    estimate_maps_images = []
    value_maps_images = []

    # Dataset aggregation
    terminal = False
    while not terminal and len(info_history) < FLAGS.max_steps_per_episode:
        obs, previous_info = env.observations()
        previous_info = copy.deepcopy(previous_info)
        optimal_action = np.argmax(exp.get_optimal_action(previous_info))

        cv2.imshow('visual', obs)
        cv2.imshow('depth', previous_info['depth'])
        cv2.waitKey(30)

        feed_dict = prepare_feed_dict(net.input_tensors, {'sequence_length': np.array([1]),
                                                          'visual_input': np.array([[observation_history[-1]]]),
                                                          'egomotion': np.array([[egomotion_history[-1]]]),
                                                          'reward': np.array([[rewards_history[-1]]]),
                                                          'estimate_map_list': estimate_maps_history[-1],
                                                          'optimal_action': np.array([optimal_action]),
                                                          'is_training': False})

        results = sess.run([net.output_tensors['action'], net.output_tensors['loss']] +
                           estimate_maps +
                           value_maps +
                           net.intermediate_tensors['estimate_map_list'], feed_dict=feed_dict)
        predict_action = np.squeeze(results[0])

        action = np.argmax(predict_action)
        obs, reward, terminal, info = env.step(action)

        cumulative_loss += results[1]
        optimal_action_history.append(np.argmax(optimal_action))
        observation_history.append(_merge_depth(obs, info['depth']))
        egomotion_history.append(environment.calculate_egomotion(previous_info['POSE'], info['POSE']))
        rewards_history.append(copy.deepcopy(reward))
        estimate_maps_history.append([tensor[:, 0, :, :, :]
                                      for tensor in results[2 + len(estimate_maps) + len(value_maps):]])
        info_history.append(copy.deepcopy(info))

        estimate_maps_images.append(results[2:2 + len(estimate_maps)])
        value_maps_images.append(results[2 + len(estimate_maps):2 + len(estimate_maps) + len(value_maps)])

    assert len(optimal_action_history) == len(observation_history) == len(egomotion_history) == len(rewards_history)

    cumulative_loss /= len(info_history)

    summary_text = ','.join('{}[{}]-{}={}'.format(key, idx, step, value)
                            for step, info in enumerate(info_history)
                            for key in ('GOAL.LOC', 'SPAWN.LOC', 'POSE', 'env_name')
                            for idx, value in enumerate(info[key]))
    step_history_summary, new_global_step = sess.run([step_history_op, update_global_step_op],
                                                     feed_dict={step_history: summary_text})
    summary_writer.add_summary(step_history_summary, global_step=np_global_step)

    summary_writer.add_summary(_build_map_summary(estimate_maps_images, value_maps_images),
                               global_step=np_global_step)
    summary_writer.add_summary(_build_trajectory_summary(cumulative_loss, rewards_history, info_history, exp),
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
    def _readout(target):
        max_axis = tf.reduce_max(target, [0, 1], keep_dims=True)
        min_axis = tf.reduce_min(target, [0, 1], keep_dims=True)
        image = (target - min_axis) / (max_axis - min_axis)
        return image

    tf.reset_default_graph()

    env = environment.get_game_environment(FLAGS.maps,
                                           multiproc=FLAGS.multiproc,
                                           random_goal=FLAGS.random_goal,
                                           random_spawn=FLAGS.random_spawn)
    exp = expert.Expert()
    net = CMAP()

    estimate_images = [_readout(estimate[0, -1, :, :, 0])
                       for estimate in net.intermediate_tensors['estimate_map_list']]
    value_images = [_readout(value[0, :, :, 0]) for value in tf.unstack(net.intermediate_tensors['value_map'], axis=1)]

    step_history = tf.placeholder(tf.string, name='step_history')
    step_history_op = tf.summary.text('game/step_history', step_history, collections=['game'])

    global_step = slim.get_or_create_global_step()
    update_global_step_op = tf.assign_add(global_step, 1)

    init_op = tf.variables_initializer([global_step])
    load_op, load_feed_dict = slim.assign_from_checkpoint(FLAGS.modeldir,
                                                          slim.get_variables_to_restore(exclude=[global_step.name]))

    init_op = tf.group(init_op, load_op)

    slim.learning.train(train_op=tf.no_op('train'),
                        logdir=FLAGS.logdir,
                        init_op=init_op,
                        init_feed_dict=load_feed_dict,
                        global_step=global_step,
                        train_step_fn=DAGGER_train_step,
                        train_step_kwargs=dict(env=env, exp=exp, net=net,
                                               update_global_step_op=update_global_step_op,
                                               step_history=step_history,
                                               step_history_op=step_history_op,
                                               estimate_maps=estimate_images,
                                               value_maps=value_images),
                        number_of_steps=FLAGS.num_games,
                        save_interval_secs=300 if not FLAGS.debug else 60,
                        save_summaries_secs=300 if not FLAGS.debug else 60)


if __name__ == '__main__':
    tf.app.run()

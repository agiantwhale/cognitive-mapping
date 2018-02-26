import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


class CMAP(object):
    def _upscale_image(self, image, scale=1):
        if scale == 0:
            return image
        estimate_size = tf.shape(image)[1:-1]
        partial_size = estimate_size / tf.constant(2 ** scale)
        crop_size = (estimate_size - partial_size) / tf.constant(2)
        crop_h, crop_w = tf.unstack(tf.cast(crop_size, dtype=tf.int32))
        image = image[:, crop_h:-crop_h, crop_w:-crop_w, :]
        image = tf.image.resize_bilinear(image, estimate_size, align_corners=True)
        return image

    @staticmethod
    def _xavier_init(num_in, num_out):
        # stddev = np.sqrt(4. / (num_in + num_out)) # xavier
        stddev = np.sqrt(1. / (num_in + num_out))  # from SELU paper
        return tf.truncated_normal_initializer(stddev=stddev)

    @staticmethod
    def _random_init(stddev=0.01):
        return tf.truncated_normal_initializer(stddev=stddev)

    def _build_model(self, m={}, estimator=None):
        is_training = self._is_training
        sequence_length = self._sequence_length
        visual_input = self._visual_input
        egomotion = self._egomotion
        reward = self._reward
        estimate_map = self._estimate_map_list
        game_size = self._game_size
        estimate_size = self._estimate_size
        estimate_scale = self._estimate_scale
        estimate_shape = self._estimate_shape
        num_iterations = self._num_iterations
        num_actions = self._num_actions
        goal_map = self._goal_map
        image_scaler = self._upscale_image
        xavier_init = self._xavier_init

        def _estimate(image):
            def _constrain_confidence(belief):
                estimate, confidence = tf.unstack(belief, axis=3)
                return tf.stack([estimate, tf.nn.sigmoid(confidence)], axis=3)

            net = image

            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                                activation_fn=tf.nn.selu,
                                biases_initializer=tf.constant_initializer(0),
                                weights_regularizer=slim.l2_regularizer(self._reg),
                                reuse=tf.AUTO_REUSE):
                last_output_channels = 4

                with slim.arg_scope([slim.conv2d],
                                    stride=1, padding='VALID'):
                    for idx, output in enumerate([(32, [7, 7]), (32, [7, 7]), (128, [3, 3]), (128, [3, 3])]):
                        channels, filter_size = output
                        scope_name = 'conv_{}x{}_{}_{}'.format(filter_size[0], filter_size[1], channels, idx)
                        net = slim.conv2d(net, channels, filter_size,
                                          scope=scope_name,
                                          weights_initializer=xavier_init(np.prod(filter_size) * last_output_channels,
                                                                          channels))
                        net = slim.max_pool2d(net, [2, 2])
                        last_output_channels = channels

                    net = slim.flatten(net)
                    last_output_channels = net.get_shape().as_list()[-1]
                    for channels in [256 * 256]:
                        net = slim.fully_connected(net, channels, scope='fc_{}'.format(channels),
                                                   weights_initializer=xavier_init(last_output_channels, channels))
                        last_output_channels = channels
                    net = tf.reshape(net, [-1, 256, 256, 1])
                    last_output_channels = 1

                with slim.arg_scope([slim.conv2d_transpose],
                                    stride=1, padding='SAME'):
                    for idx, channels in enumerate([32, 16, 2]):
                        filter_size = [3, 3]
                        scope_name = 'deconv_{}x{}_{}_{}'.format(filter_size[0], filter_size[1], channels, idx)
                        initializer = xavier_init(last_output_channels, np.prod(filter_size) * channels)
                        net = slim.conv2d_transpose(net, channels, filter_size,
                                                    scope=scope_name, weights_initializer=initializer)
                        last_output_channels = channels

                    beliefs = [self._upscale_image(net, i) for i in xrange(estimate_scale)]

            return [_constrain_confidence(belief) for belief in beliefs]

        def _apply_egomotion(tensor, scale_index, ego):
            tx, ty, rotation = tf.unstack(ego, axis=1)

            scale = tf.constant((2 ** scale_index) / (game_size / float(estimate_size)), dtype=tf.float32)

            rot_mat = tf.contrib.image.angles_to_projective_transforms(tf.negative(rotation),
                                                                       estimate_size, estimate_size)

            zero = tf.zeros_like(tx)
            ones = tf.ones_like(tx)
            trans_mat = tf.stack([ones, zero, tf.multiply(tx, scale)] +
                                 [zero, ones, tf.multiply(tf.negative(ty), scale)] +
                                 [zero, zero], axis=1)

            transform = tf.contrib.image.compose_transforms(rot_mat, trans_mat)

            return tf.contrib.image.transform(tensor, transform)

        def _delta_reward_map(reward):
            h, w, c = estimate_shape
            m_h, m_w = int((h - 1) / 2), int((w - 1) / 2)

            return tf.pad(tf.expand_dims(reward, axis=2),
                          tf.constant([[0, 0], [m_h - 1, w - m_h], [m_w - 1, w - m_w]]))

        def _warp(temp_belief, prev_belief):
            temp_estimate, temp_confidence, temp_rewards = tf.unstack(temp_belief, axis=3)
            prev_estimate, prev_confidence, prev_rewards = tf.unstack(prev_belief, axis=3)

            current_confidence = temp_confidence + prev_confidence
            current_estimate = tf.divide(tf.multiply(temp_estimate, temp_confidence) +
                                         tf.multiply(prev_estimate, prev_confidence),
                                         current_confidence)
            current_rewards = temp_rewards + prev_rewards
            current_belief = tf.stack([current_estimate, current_confidence, current_rewards], axis=3)
            return current_belief

        def _scale_belief(belief, scale):
            last_output_channels = 2
            net = belief
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.selu,
                                biases_initializer=None if not self._biased_fuser else self._random_init(),
                                weights_regularizer=slim.l2_regularizer(self._reg),
                                stride=1, padding='SAME', reuse=tf.AUTO_REUSE):
                for idx, channels in enumerate([2, 1]):
                    scope = 'scaler_{}_{}'.format(channels, idx)
                    if not self._unified_fuser:
                        scope = '{}_{}'.format(scope, scale)
                    net = slim.conv2d(net, channels, [1, 1], scope=scope,
                                      weights_initializer=self._xavier_init(last_output_channels, channels))
                    last_output_channels = channels

            return tf.image.resize_bilinear(net, tf.constant([self._vin_size, self._vin_size]), align_corners=True)

        def _fuse_belief(belief, scale):
            last_output_channels = 2
            net = belief
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.selu,
                                biases_initializer=None if not self._biased_fuser else self._random_init(),
                                weights_regularizer=slim.l2_regularizer(self._reg),
                                stride=1, padding='SAME', reuse=tf.AUTO_REUSE):
                for idx, channels in enumerate([2, 1]):
                    scope = 'fuser_{}_{}'.format(channels, idx)
                    if not self._unified_fuser:
                        scope = '{}_{}'.format(scope, scale)
                    net = slim.conv2d(net, channels, [1, 1], scope=scope,
                                      weights_initializer=self._xavier_init(last_output_channels, channels))
                    last_output_channels = channels

                return net

        def _vin(rewards_map, scale):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                weights_initializer=self._xavier_init(2 * 3 * 3, num_actions),
                                biases_initializer=None if not self._biased_vin else self._random_init(),
                                weights_regularizer=slim.l2_regularizer(self._reg),
                                reuse=tf.AUTO_REUSE):
                scope = 'VIN_actions'
                if not self._unified_vin:
                    scope = '{}_{}'.format(scope, scale)

                actions_map = slim.conv2d(rewards_map, num_actions, [3, 3], scope='{}_initial'.format(scope),
                                          weights_initializer=self._xavier_init(1 * 3 * 3, num_actions))
                values_map = tf.reduce_max(actions_map, axis=3, keep_dims=True)

                for i in xrange(num_iterations - 1):
                    rv = tf.concat([rewards_map, values_map], axis=3)
                    actions_map = slim.conv2d(rv, num_actions, [3, 3], scope=scope)
                    values_map = tf.reduce_max(actions_map, axis=3, keep_dims=True)

                return values_map, actions_map

        class BiLinearSamplingCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return [tf.TensorShape(estimate_shape)] * estimate_scale

            @property
            def output_size(self):
                return [tf.TensorShape((estimate_size, estimate_size, 2))] * estimate_scale + \
                       [tf.TensorShape((estimate_size, estimate_size, 3))] * estimate_scale + \
                       [tf.TensorShape((estimate_size, estimate_size, 1))] * estimate_scale

            def __call__(self, inputs, state, scope=None):
                image, goal, ego, re = inputs

                delta_reward_map = tf.expand_dims(_delta_reward_map(re), axis=3)

                current_scaled_estimates = _estimate(image) if estimator is None else estimator(image)
                current_scaled_estimates = [tf.concat([estimate, delta_reward_map], axis=3)
                                            for estimate in current_scaled_estimates]
                previous_scaled_estimates = [_apply_egomotion(belief, scale_index, ego)
                                             for scale_index, belief in enumerate(state)]
                scaled_estimates = [_warp(c, p) for c, p in zip(current_scaled_estimates, previous_scaled_estimates)]
                scaled_goal_maps = [image_scaler(goal, idx) for idx in xrange(estimate_scale)]

                merged_belief = [tf.concat([goal, tf.expand_dims(belief[:, :, :, 0], axis=3)], axis=3)
                                 for goal, belief in zip(scaled_goal_maps, scaled_estimates)]

                output = merged_belief + scaled_estimates + scaled_goal_maps

                return output, scaled_estimates

        with tf.variable_scope('mapper'):
            normalized_input = slim.batch_norm(visual_input, is_training=is_training, scope='visual/batch_norm')
            normalized_goal = slim.batch_norm(goal_map, is_training=is_training, scope='goal/batch_norm')
            bilinear_cell = BiLinearSamplingCell()
            results, final_belief = tf.nn.dynamic_rnn(bilinear_cell,
                                                      (normalized_input,
                                                       normalized_goal,
                                                       egomotion,
                                                       tf.expand_dims(reward, axis=2)),
                                                      sequence_length=sequence_length,
                                                      initial_state=estimate_map,
                                                      swap_memory=True)

            scaled_merged_beliefs = results[:estimate_scale]
            results = results[estimate_scale:]

            m['estimate_map_list'] = results[:estimate_scale]
            results = results[estimate_scale:]

            m['goal_map_list'] = results[:estimate_scale]
            results = results[estimate_scale:]

            assert len(results) == 0

        with tf.variable_scope('planner'):
            batch_size, timesteps, w, h, channels = tf.unstack(tf.shape(scaled_merged_beliefs[0]))

            roll_time = lambda x: tf.reshape(x, tf.concat([[batch_size, timesteps], tf.shape(x)[1:]], axis=0))
            unroll_time = lambda x: tf.reshape(x, tf.concat([[batch_size * timesteps], tf.shape(x)[2:]], axis=0))
            merged_belief = [unroll_time(maps) for idx, maps in enumerate(scaled_merged_beliefs)]

            rewards = []
            values = []
            actions = []

            values_map = None
            for idx, belief in enumerate(merged_belief):
                rewards_map = _scale_belief(belief, idx)
                if values_map is not None:
                    rewards_map = _fuse_belief(tf.concat([rewards_map, image_scaler(values_map)], axis=3), idx)
                values_map, actions_map = _vin(rewards_map, idx)

                rewards.append(rewards_map)
                values.append(values_map)
                actions.append(actions_map)

            if self._flatten_action:
                net = slim.flatten(values_map)
                output_channels = net.get_shape().as_list()[-1]
                net = slim.fully_connected(net, 64,
                                           reuse=tf.AUTO_REUSE,
                                           activation_fn=tf.nn.selu,
                                           weights_initializer=self._xavier_init(output_channels, 64),
                                           biases_initializer=tf.zeros_initializer(),
                                           weights_regularizer=slim.l2_regularizer(self._reg),
                                           scope='logits_64')
                predictions = slim.fully_connected(net, num_actions,
                                                   reuse=tf.AUTO_REUSE,
                                                   activation_fn=None,
                                                   weights_initializer=self._xavier_init(64, num_actions),
                                                   biases_initializer=tf.zeros_initializer(),
                                                   weights_regularizer=slim.l2_regularizer(self._reg),
                                                   scope='logits')
            else:
                center = int(self._vin_size / 2)
                predictions = slim.flatten(actions_map[:, center, center, :])

            m['unrolled_predictions'] = predictions
            m['predictions'] = roll_time(predictions)
            m['reward_map_list'] = [roll_time(reward) for reward in rewards]
            m['value_map_list'] = [roll_time(value) for value in values]
            m['action_map_list'] = [roll_time(action) for action in actions]

        return m['predictions'][:, -1, :]

    def __init__(self, image_size=(84, 84, 4), game_size=1280, estimate_size=256, estimate_scale=3,
                 estimator=None, num_actions=4, num_iterations=10, vin_size=16, flatten_action=True,
                 unified_fuser=True, unified_vin=True,
                 biased_fuser=False, biased_vin=False,
                 regularization=0.):
        self._image_size = image_size
        self._game_size = game_size
        self._estimate_size = estimate_size
        self._estimate_shape = (estimate_size, estimate_size, 3)
        self._estimate_scale = estimate_scale
        self._num_actions = num_actions
        self._num_iterations = num_iterations
        self._vin_size = vin_size
        self._flatten_action = flatten_action
        self._reg = regularization
        self._unified_fuser = unified_fuser
        self._unified_vin = unified_vin
        self._biased_fuser = biased_fuser
        self._biased_vin = biased_vin
        self._is_training = tf.placeholder(tf.bool, name='is_training')

        self._sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self._visual_input = tf.placeholder(tf.float32, [None, None] + list(self._image_size),
                                            name='visual_input')
        self._egomotion = tf.placeholder(tf.float32, (None, None, 3), name='egomotion')
        self._reward = tf.placeholder(tf.float32, (None, None), name='reward')
        self._goal_map = tf.placeholder(tf.float32, (None, None, estimate_size, estimate_size, 1), name='goal_map')
        self._estimate_map_list = [tf.placeholder(tf.float32, (None, estimate_size, estimate_size, 3),
                                                  name='estimate_map_{}'.format(i))
                                   for i in xrange(estimate_scale)]
        self._optimal_action = tf.placeholder(tf.int32, [None, None], name='optimal_action')
        self._optimal_estimate = tf.placeholder(tf.int32, [None, None, estimate_size, estimate_size],
                                                name='optimal_estimate')

        tensors = {}
        logits = self._build_model(tensors, estimator=estimator)

        self._action = tf.nn.softmax(logits)

        reshaped_optimal_action = tf.reshape(self._optimal_action, [-1])
        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=reshaped_optimal_action,
                                                            logits=tensors['unrolled_predictions'])
        self._loss += sum(tf.losses.get_regularization_loss('{}/.*'.format(scope_name))
                          for scope_name in 'mapper planner'.split())

        reshaped_estimate_map = tf.reshape(tensors['estimate_map_list'][0][:, :, :, :, 0],
                                           [-1, estimate_size, estimate_size])
        reshaped_optimal_estimate_map = tf.reshape(self._optimal_estimate, [-1, estimate_size, estimate_size])
        self._prediction_loss = tf.losses.mean_squared_error(reshaped_optimal_estimate_map, reshaped_estimate_map)
        self._prediction_loss += tf.losses.get_regularization_loss('mapper/.*')

        self._intermediate_tensors = tensors

    @property
    def input_tensors(self):
        return {
            'is_training': self._is_training,
            'sequence_length': self._sequence_length,
            'visual_input': self._visual_input,
            'egomotion': self._egomotion,
            'reward': self._reward,
            'goal_map': self._goal_map,
            'estimate_map_list': self._estimate_map_list,
            'optimal_action': self._optimal_action,
            'optimal_estimate': self._optimal_estimate
        }

    @property
    def intermediate_tensors(self):
        return self._intermediate_tensors

    @property
    def output_tensors(self):
        return {
            'action': self._action,
            'loss': self._loss,
            'estimate_loss': self._prediction_loss,
        }


if __name__ == "__main__":
    CMAP()

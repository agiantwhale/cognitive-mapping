import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


class CMAP(object):
    def _upscale_image(self, image, scale=1):
        if scale == 0:
            return image
        estimate_size = self._estimate_size
        partial_size = int(estimate_size / (2 ** scale))
        crop_size = (estimate_size - partial_size) / 2
        image = image[:, crop_size:-crop_size, crop_size:-crop_size, :]
        image = tf.image.resize_bilinear(image, tf.constant([estimate_size, estimate_size]),
                                         align_corners=True)
        return image

    def _build_mapper(self, m={}, estimator=None):
        is_training = self._is_training
        sequence_length = self._sequence_length
        visual_input = self._visual_input
        egomotion = self._egomotion
        reward = self._reward
        estimate_map = self._estimate_map_list
        estimate_size = self._estimate_size
        estimate_scale = self._estimate_scale
        estimate_shape = self._estimate_shape
        goal_map = self._goal_map

        def _estimate(image):
            def _xavier_init(num_in, num_out):
                stddev = np.sqrt(4. / (num_in + num_out))
                return tf.truncated_normal_initializer(stddev=stddev)

            def _constrain_confidence(belief):
                estimate, confidence = tf.unstack(belief, axis=3)
                return tf.stack([estimate, tf.nn.sigmoid(confidence)], axis=3)

            beliefs = []
            net = image

            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                                activation_fn=tf.nn.elu,
                                biases_initializer=tf.constant_initializer(0),
                                reuse=tf.AUTO_REUSE):
                last_output_channels = 4

                with slim.arg_scope([slim.conv2d],
                                    stride=1, padding='SAME'):
                    net = slim.conv2d(net, 32, [7, 7], scope='mapper/conv_7x7_64',
                                      weights_initializer=_xavier_init(np.prod(7) * last_output_channels, 64))
                    net = slim.max_pool2d(net, [2, 2])
                    last_output_channels = 32

                    for output in [(64, [3, 3]), (128, [3, 3])]:
                        channels, filter_size = output
                        scope = 'mapper/conv_{}x{}_{}_'.format(filter_size[0], filter_size[1], channels)
                        net = slim.conv2d(net, channels, filter_size, scope=scope + '1',
                                          weights_initializer=_xavier_init(np.prod(filter_size) * last_output_channels,
                                                                           channels))
                        residual = net
                        net = slim.conv2d(net, channels, filter_size, scope=scope + '2',
                                          weights_initializer=_xavier_init(np.prod(filter_size) * last_output_channels,
                                                                           channels))
                        net = net + residual
                        last_output_channels = channels

                    net = slim.flatten(net)
                    for channels in [1024, 4096]:
                        net = slim.fully_connected(net, channels, scope='mapper/fc_{}'.format(channels),
                                                   weights_initializer=_xavier_init(42 * 42 * 128, channels))
                        last_output_channels = channels
                    net = tf.reshape(net, [-1, estimate_size, estimate_size, 1])

                with slim.arg_scope([slim.conv2d_transpose],
                                    stride=1, padding='SAME'):
                    for channels in [32, 16, 2]:
                        net = slim.conv2d_transpose(net, channels, [7, 7],
                                                    scope='mapper/deconv_7x7_{}'.format(channels),
                                                    weights_initializer=_xavier_init(7 * 7 * last_output_channels,
                                                                                     channels))
                        last_output_channels = channels

                    beliefs = [self._upscale_image(net, i) for i in xrange(estimate_scale)]

            return [_constrain_confidence(belief) for belief in beliefs]

        def _apply_egomotion(tensor, scale_index, ego):
            translation, rotation = tf.unstack(ego, axis=1)

            scale = tf.constant((2 ** scale_index) / (1400. / self._estimate_size), dtype=tf.float32)

            rot_op = tf.contrib.image.angles_to_projective_transforms(tf.negative(rotation),
                                                                      estimate_size, estimate_size)

            zero = tf.zeros_like(translation)
            ones = tf.ones_like(translation)
            trans_op = tf.stack([ones, zero, tf.multiply(translation, scale)] +
                                [zero, ones, zero, zero, zero], axis=1)

            transform = tf.contrib.image.compose_transforms(rot_op, trans_op)

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

        class BiLinearSamplingCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return [tf.TensorShape(estimate_shape)] * estimate_scale

            @property
            def output_size(self):
                return self.state_size

            def __call__(self, inputs, state, scope=None):
                image, ego, re = inputs

                delta_reward_map = tf.expand_dims(_delta_reward_map(re), axis=3)

                current_scaled_estimates = _estimate(image) if estimator is None else estimator(image)
                current_scaled_estimates = [tf.concat([estimate, delta_reward_map], axis=3)
                                            for estimate in current_scaled_estimates]
                previous_scaled_estimates = [_apply_egomotion(belief, scale_index, ego)
                                             for scale_index, belief in enumerate(state)]
                outputs = [_warp(c, p) for c, p in zip(current_scaled_estimates, previous_scaled_estimates)]

                return outputs, outputs

        normalized_input = slim.batch_norm(visual_input, is_training=is_training, scope='mapper/input/batch_norm')
        bilinear_cell = BiLinearSamplingCell()
        interm_beliefs, final_belief = tf.nn.dynamic_rnn(bilinear_cell,
                                                         (normalized_input, egomotion, tf.expand_dims(reward, axis=2)),
                                                         sequence_length=sequence_length,
                                                         initial_state=estimate_map,
                                                         swap_memory=True)

        goal_map = tf.expand_dims(goal_map, axis=3)
        scaled_goal_maps = [self._upscale_image(goal_map, idx) for idx in xrange(len(final_belief))]
        norm_belief = [slim.batch_norm(belief[:, :, :, 0], reuse=tf.AUTO_REUSE,
                                       is_training=is_training,
                                       scope='mapper/estimate/batch_norm')
                       for belief in final_belief]
        final_belief = [tf.concat([goal, tf.expand_dims(belief, axis=3)], axis=3)
                        for goal, belief in zip(scaled_goal_maps, norm_belief)]

        m['estimate_map_list'] = interm_beliefs
        m['goal_map_list'] = scaled_goal_maps

        return final_belief

    def _build_planner(self, scaled_beliefs, m={}):
        is_training = self._is_training
        batch_size = tf.shape(scaled_beliefs[0])[0]
        image_scaler = self._upscale_image
        estimate_size = self._estimate_size
        value_map_size = (estimate_size, estimate_size, 1)
        num_actions = self._num_actions
        num_iterations = self._num_iterations

        def _fuse_belief(belief):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.elu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=1.15),
                                biases_initializer=None,
                                stride=1, padding='SAME', reuse=tf.AUTO_REUSE):
                net = slim.conv2d(belief, 1, [1, 1], scope='planner/fuser')
                return net

        class HierarchicalVINCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return tf.TensorShape(value_map_size)

            @property
            def output_size(self):
                return [tf.TensorShape(value_map_size),
                        tf.TensorShape(value_map_size),
                        tf.TensorShape((estimate_size, estimate_size, num_actions))]

            def __call__(self, inputs, state, scope=None):
                # Upscale previous value map
                state = image_scaler(state)

                rewards_map = _fuse_belief(tf.concat([inputs, state], axis=3))

                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.42),
                                    biases_initializer=None,
                                    reuse=tf.AUTO_REUSE):
                    actions_map = slim.conv2d(rewards_map, num_actions, [3, 3],
                                              scope='planner/VIN_actions_initial')
                    values_map = tf.reduce_max(actions_map, axis=3, keep_dims=True)

                    for i in xrange(num_iterations - 1):
                        rv = tf.concat([rewards_map, values_map], axis=3)
                        actions_map = slim.conv2d(rv, num_actions, [3, 3],
                                                  scope='planner/VIN_actions')
                        values_map = tf.reduce_max(actions_map, axis=3, keep_dims=True)

                return [rewards_map, values_map, actions_map], values_map

        beliefs = tf.stack(scaled_beliefs, axis=1)
        vin_cell = HierarchicalVINCell()
        interm_values_map, final_values_map = tf.nn.dynamic_rnn(vin_cell, beliefs,
                                                                initial_state=vin_cell.zero_state(batch_size,
                                                                                                  tf.float32),
                                                                swap_memory=True)
        m['fused_map'] = interm_values_map[0]
        m['value_map'] = interm_values_map[1]

        values_features = interm_values_map[-1][:, -1, estimate_size / 2, estimate_size / 2, :]
        actions_logit = slim.fully_connected(values_features, num_actions,
                                             activation_fn=None,
                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.7),
                                             biases_initializer=None,
                                             scope='planner/logits')

        return actions_logit

    def __init__(self, image_size=(84, 84, 4), estimate_size=64, estimate_scale=3,
                 estimator=None, num_actions=4, num_iterations=10):
        self._image_size = image_size
        self._estimate_size = estimate_size
        self._estimate_shape = (estimate_size, estimate_size, 3)
        self._estimate_scale = estimate_scale
        self._num_actions = num_actions
        self._num_iterations = num_iterations
        self._is_training = tf.placeholder(tf.bool, name='is_training')

        self._sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self._visual_input = tf.placeholder(tf.float32, [None, None] + list(self._image_size),
                                            name='visual_input')
        self._egomotion = tf.placeholder(tf.float32, (None, None, 2), name='egomotion')
        self._reward = tf.placeholder(tf.float32, (None, None), name='reward')
        self._goal_map = tf.placeholder(tf.float32, (None, estimate_size, estimate_size), name='goal_map')
        self._estimate_map_list = [tf.placeholder(tf.float32, (None, estimate_size, estimate_size, 3),
                                                  name='estimate_map_{}'.format(i))
                                   for i in xrange(estimate_scale)]
        self._optimal_action = tf.placeholder(tf.int32, [None], name='optimal_action')

        tensors = {}
        scaled_beliefs = self._build_mapper(tensors, estimator=estimator)
        unscaled_action = self._build_planner(scaled_beliefs, tensors)

        self._action = tf.nn.softmax(unscaled_action)
        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self._optimal_action, logits=unscaled_action)
        self._loss += tf.losses.get_regularization_loss()

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
            'optimal_action': self._optimal_action
        }

    @property
    def intermediate_tensors(self):
        return self._intermediate_tensors

    @property
    def output_tensors(self):
        return {
            'action': self._action,
            'loss': self._loss
        }


if __name__ == "__main__":
    CMAP()

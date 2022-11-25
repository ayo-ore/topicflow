from abc import abstractmethod
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import activations, layers, metrics
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm


def make_dnn(
    hidden_units,
    output_dim,
    activation='relu',
    output_activation=None,
    initializer='glorot_uniform',
    constraint=None,
    regularizer=None,
    dropout=None
    ):

    hidden_layers = []
    for i, s in enumerate(hidden_units):
        hidden_layers.append(
            layers.Dense(
                s, activation=activation, kernel_initializer=initializer,
                kernel_regularizer=regularizer, kernel_constraint=constraint,
                name=f"dense_{i}"
            )
        )
        if dropout is not None:
            hidden_layers.append(layers.Dropout(dropout, name=f'dropout_{i}'))
    hidden_layers.append(
        layers.Dense(
            output_dim, activation=output_activation,
            kernel_initializer=initializer, kernel_regularizer=regularizer,
            kernel_constraint=constraint, name=f"output"
        )
    )
    dnn = Sequential(hidden_layers)
    return dnn
    

class ResidualBlock(tf.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers,
        activation='relu',
        first=False,
        dropout=None,
        batchnorm=False,
        regularizer=None
    ):

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.first = first
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.regularizer = regularizer
        self.build()

    def build(self):
        if self.first:
            self.embedding = layers.Dense(
                self.hidden_dim, activation=self.activation,
                kernel_regularizer=self.regularizer
            )

        self.dense_layers = [
            layers.Dense(self.hidden_dim, kernel_regularizer=self.regularizer)
            for _ in range(self.num_layers)
        ]
        self.activation_layer = layers.Activation(self.activation)

        if self.batchnorm:
            self.batchnorms = [layers.BatchNormalization()] * self.num_layers

        if self.dropout is not None:
            self.dropout_layer = layers.Dropout(self.dropout)

    @tf.function
    def __call__(self, x):
        if self.first:
            x = self.embedding(x)
        for i, layer in enumerate(self.dense_layers):
            y = layer(x if i == 0 else y)
            if self.dropout:
                y = self.dropout_layer(y)
            if self.batchnorm:
                y = self.batchnorms[i](y)
            y = self.activation_layer(y)

        return x + y


class ResNet(Model):

    def __init__(
        self,
        num_blocks,
        hidden_dim,
        num_layers,
        output_dim,
        activation='tanh',
        batchnorm=False,
        dropout=None,
        regularizer=None,
        *args,
        **kwargs
    ):

        super(ResNet, self).__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.regularizer = regularizer

        self.blocks = [
            ResidualBlock(
                self.hidden_dim, self.num_layers, first=i == 0,
                activation=self.activation, batchnorm=self.batchnorm,
                regularizer=self.regularizer
            ) for i in range(self.num_blocks)
        ]
        self.final = layers.Dense(self.output_dim, activation=None,
                                  kernel_regularizer=self.regularizer)

    @tf.function
    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)


class Flow(Model):

    def __init__(
        self,
        dim: int,
        num_transform_layers: int,
        bijector_config:dict,
        net_config:dict,
        batchnorm:bool=True,
        training:bool=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.dim = dim
        self.num_transform_layers = num_transform_layers
        self.base = tfd.Sample(tfd.Normal(0, 1), sample_shape=[self.dim])
        self.nll_tracker = metrics.Mean(name="NegativeLogLikelihood")
        self.net_config = net_config
        self.bijector_config = bijector_config
        self.batchnorm = batchnorm
        self.training = training
        self.build()

    @abstractmethod
    def build_param_network(self):
        pass

    @abstractmethod
    def build_transform_layer(self, param_net):
        pass

    @property
    def input_shape(self):
        return [None, self.dim]

    def build(self):

        # build transformation layers as defined by child class
        self.param_networks = [
            self.build_param_network()
            for _ in range(self.num_transform_layers)
        ]
        self.transform_layers = [
            self.build_transform_layer(param_net)
            for param_net in self.param_networks
        ]

        # create bijector chain
        self.links = [self.transform_layers.pop(0)]
        self.lus, self.ps = [], []

        # join multiple flow steps with invertible linear transformation and
        # batchnorm
        for transform_layer in self.transform_layers:
            random_matrix = tf.random.uniform(
                (self.dim, self.dim), dtype=tf.float32
            )
            random_orthonormal = tf.linalg.qr(random_matrix)[0]
            lu, p = tf.linalg.lu(random_orthonormal)
            self.lus.append(tf.Variable(lu, trainable=True, name='LU'))
            self.ps.append(tf.Variable(p, trainable=False, name='P'))
            self.links.append(tfb.ScaleMatvecLU(self.lus[-1], self.ps[-1]))
            if self.batchnorm:
                self.links.append(tfb.BatchNormalization(training=self.training))
            self.links.append(transform_layer)
        self.bijector = tfb.Chain(self.links)
        self.flow = tfd.TransformedDistribution(
            distribution=self.base, bijector=self.bijector
        )

        # register that the model is built
        super().build(self.input_shape)

    def sample(self, num_samples):
        return self.flow.sample(num_samples)

    @tf.function
    def call(self, x):
        return self.flow.log_prob(x)

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            nll = -tf.reduce_mean(self(batch))
        grads = tape.gradient(nll, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.nll_tracker.update_state(nll)
        return {self.nll_tracker.name: self.nll_tracker.result()}

    @tf.function
    def test_step(self, batch):
        nll = -tf.reduce_mean(self(batch))
        self.nll_tracker.update_state(nll)
        return {self.nll_tracker.name: self.nll_tracker.result()}

    @property
    def metrics(self):
        return [self.nll_tracker]


class ConditionalFlow(Flow):

    def __init__(self, condition_dim, *args, **kwargs):

        self.condition_dim = condition_dim
        super().__init__(*args, **kwargs)

    @property
    def input_shape(self):
        return [[None, self.dim], [None, self.condition_dim]]

    @abstractmethod
    def _condition_kwargs(self, c):
        pass

    def sample(self, num_samples, c):
        c = tf.repeat([c], num_samples, axis=0)
        return self.flow.sample(num_samples, bijector_kwargs=self._condition_kwargs(c))

    @tf.function
    def call(self, x):
        return self.flow.log_prob(x[0], bijector_kwargs=self._condition_kwargs(x[1]))

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            nll = -tf.reduce_mean(self(batch))
        grads = tape.gradient(nll, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.nll_tracker.update_state(nll)
        return {self.nll_tracker.name: self.nll_tracker.result()}

    @tf.function
    def test_step(self, batch):
        nll = -tf.reduce_mean(self(batch))
        self.nll_tracker.update_state(nll)
        return {self.nll_tracker.name: self.nll_tracker.result()}


class AffineCouplingFlow(Flow):

    def build_param_network(self):
        return AffineNet(input_dim=self.dim//2, **self.net_config)

    def build_transform_layer(self, affine_net):
        return tfb.RealNVP(shift_and_log_scale_fn=affine_net, **self.bijector_config)


class ConditionalAffineCouplingFlow(ConditionalFlow):

    def build_param_network(self):
        return ConditionalAffineNet(
            input_dim=self.dim//2, condition_dim=self.condition_dim, **self.net_config
        )

    def build_transform_layer(self, affine_net):
        return tfb.RealNVP(shift_and_log_scale_fn=affine_net, **self.bijector_config)

    def _condition_kwargs(self, c):
        return {'real_nvp': {'cond': c}}

class RQSCouplingFlow(Flow):

    def build_param_network(self):
        return RQSNet(dim=self.dim, **self.net_config)

    def build_transform_layer(self, affine_net):
        return tfb.RealNVP(shift_and_log_scale_fn=affine_net, **self.bijector_config)


class ConditionalRQSCouplingFlow(ConditionalFlow):

    def build_param_network(self):
        return ConditionalRQSNet(
            dim=self.dim, condition_dim=self.condition_dim, **self.net_config
        )

    def build_transform_layer(self, affine_net):
        return tfb.RealNVP(shift_and_log_scale_fn=affine_net, **self.bijector_config)

    def _condition_kwargs(self, c):
        return {'real_nvp': {'cond': c}}

class AffineInverseAutoregressiveFlow(Flow):

    def soft_clamp(self, shift_and_logscale_fn):
        @tf.function
        def soft_clamped(x):
            x = shift_and_logscale_fn(x)
            shift, log_scale = tf.split(x, 2, axis=-1)
            return tf.concat([shift, 2*tf.math.tanh(log_scale)], axis=-1)
        return soft_clamped
        
    def build_param_network(self):
        return tfb.AutoregressiveNetwork(
            params=2, event_shape=[self.dim], activation='relu', **self.transform_config
        )

    def build_transform_layer(self, param_net):
        return tfb.Invert(tfb.MaskedAutoregressiveFlow(self.soft_clamp(param_net)))

class AffineNet(tf.Module):

    def __init__(self, input_dim, layer_sizes, shift_only=False, regularizer=None, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.shift_only = shift_only
        self.regularizer = regularizer
        self.build()

    @property
    def input_layers(self):
        return [Input(self.input_dim, name="input")]

    @tf.Module.with_name_scope
    def build(self):
        inp = self.input_layers
        x = tf.concat(inp, axis=-1)

        # dense layers
        for i, s in enumerate(self.layer_sizes):
            x = layers.Dense(
                s, name=f"layer_{i}", activation='relu', kernel_regularizer=self.regularizer
            )(x)

        # output
        x = layers.Dense(
            2 - int(self.shift_only), name="output", activation='linear',
            kernel_regularizer=self.regularizer
        )(x)

        if self.shift_only:
            out = x, None
        else:
            shift, log_scale = tf.split(x, 2, axis=-1)
            log_scale = tf.multiply(tf.math.tanh(log_scale), 2)  # 'soft clamping'
            out = shift, log_scale
        self.network = Model(inputs=inp, outputs=out)

    @tf.function
    def __call__(self, x, input_dim):
        return self.network(x)


class ConditionalAffineNet(AffineNet):

    def __init__(self, condition_dim, *args, **kwargs):
        self.condition_dim = condition_dim
        super().__init__(*args, **kwargs)

    @property
    def input_layers(self):
        return [Input(self.input_dim, name="input"),
                Input(self.condition_dim, name="conditional_input")]

    @tf.function
    def __call__(self, x, input_dim, cond):
        return self.network([x, cond])


class RQSNet(tf.Module):

    def __init__(
        self, dim, layer_sizes, num_knots, boundary, min_width, min_slope,
        regularizer=None, dropout=None, **kwargs
    ):

        super().__init__(**kwargs)
        # split in-out dimensions since tensorflow_probability
        # implementation conditions x[d+1:D] on x[1:d] for autoregressive-like model
        self.input_dim = dim//2
        self.output_dim = dim - dim//2
        self.layer_sizes = layer_sizes
        self.num_knots = num_knots
        self.boundary = boundary
        self.regularizer = regularizer
        self.dropout = dropout
        self.min_width = min_width
        self.min_slope = min_slope
        self.build()

    @property
    def input_layers(self):
        return [Input(self.input_dim, name="input")]

    @property
    def num_bins(self):
        return self.num_knots - 1
    
    @tf.function
    def _normalize_bins(self, x):
        norm = self.boundary[1] - self.boundary[0] - self.num_bins * self.min_width
        return activations.softmax(x, axis=-1) * norm + self.min_width
    
    @tf.function
    def _normalize_slopes(self, x):
        return activations.softplus(x) + self.min_slope

    @tf.Module.with_name_scope
    def build(self):

        # inputs
        inp = self.input_layers
        x = tf.concat(inp, axis=-1)
        # hidden layers
        for i, d in enumerate(self.layer_sizes):
            x = layers.Dense(
                d, name=f"layer_{i}", activation='relu', kernel_regularizer=self.regularizer
            )(x)
            if self.dropout:
                x = layers.Dropout(self.dropout, name=f"dropout_{i}")(x)
        # outputs
        x = layers.Dense(
            (3 * self.num_bins - 1) * self.output_dim, name=f"output", activation='linear',
            kernel_regularizer=self.regularizer
        )(x)
        x = tf.reshape(x, tf.concat((tf.shape(x)[:-1], (self.output_dim, -1)), axis=0))

        # split and normalize widths, heights, slopes for spline
        w, h, s = tf.split(x, (self.num_bins, self.num_bins, self.num_bins-1), -1)
        w = layers.Lambda(self._normalize_bins)(w)
        h = layers.Lambda(self._normalize_bins)(h)
        s = layers.Lambda(self._normalize_slopes)(s)

        out = w, h, s
        self.network = Model(
            inputs=inp, outputs=out, name='spline_params_network'
        )

    @tf.function
    def __call__(self, x, input_dim):
        return tfb.RationalQuadraticSpline(
            *self.network(x), range_min=self.boundary[0]
        )


class ConditionalRQSNet(RQSNet):

    def __init__(self, condition_dim, *args, **kwargs):
        self.condition_dim = condition_dim
        super().__init__(*args, **kwargs)

    @property
    def input_layers(self):
        return [Input(self.input_dim, name="input"),
                Input(self.condition_dim, name="conditional_input")]

    @tf.function
    def __call__(self, x, input_dim, cond):
        return tfb.RationalQuadraticSpline(
            *self.network([x, cond]), range_min=self.boundary[0]
        )

class ODENet(tf.Module):

    def __init__(
        self,
        input_dim,
        num_residual_blocks,
        hidden_dim,
        num_block_layers,
        activation,
        regularizer,
        dropout,
        *args,
        **kwargs
        ):

        super(ODENet, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.num_residual_blocks = num_residual_blocks
        self.hidden_dim = hidden_dim
        self.num_block_layers = num_block_layers
        self.input_dim = input_dim
        self.activation = activation
        self.regularizer = regularizer
        self.dropout = dropout
        self.network = ResNet(
            num_blocks=self.num_residual_blocks,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_block_layers,
            output_dim=self.input_dim,
            activation=self.activation,
            regularizer=self.regularizer,
            dropout=self.dropout
        )

    @tf.function
    def __call__(self, t, x):
        t = tf.expand_dims(tf.repeat(t, tf.shape(x)[0]), axis=1)
        inp = tf.concat([t, x], axis=-1)
        return self.network(inp)

    def summary(self):
        self.network.summary()


class ConditionalODENet(ODENet):

    def __init__(self, condition_dim, *args, **kwargs):
        self.condition_dim = condition_dim
        super().__init__(*args, **kwargs)

    def __call__(self, t, x, cond):
        t = tf.expand_dims(tf.repeat(t, tf.shape(x)[0]), axis=1)
        inp = tf.concat([t, x, cond], axis=-1)
        return self.network(inp)


class NeuralODEFlow(Flow):

    def build_param_network(self):
        return ODENet(input_dim=self.dim, **self.net_config)

    def build_transform_layer(self, ode_net):
        solver = tfm.ode.DormandPrince(atol=self.bijector_config['atol'])
        return tfb.FFJORD(
            state_time_derivative_fn=ode_net, ode_solve_fn=solver.solve,
            final_time=self.bijector_config['final_time'],
            trace_augmentation_fn=(
                tfb.ffjord.trace_jacobian_exact if
                self.bijector_config['exact_trace']
                else tfb.ffjord.trace_jacobian_hutchinson
            )
        )


class ConditionalNeuralODEFlow(ConditionalFlow):

    def build_param_network(self):
        return ConditionalODENet(
            input_dim=self.dim, condition_dim=self.condition_dim,
            **self.net_config
        )

    def build_transform_layer(self, ode_net):
        solver = tfm.ode.DormandPrince(atol=self.bijector_config['atol'])
        return tfb.FFJORD(
            state_time_derivative_fn=ode_net, ode_solve_fn=solver.solve,
            final_time=self.bijector_config['final_time'],
            trace_augmentation_fn=(
                tfb.ffjord.trace_jacobian_exact if
                self.bijector_config['exact_trace']
                else tfb.ffjord.trace_jacobian_hutchinson
            )
        )

    def _condition_kwargs(self, c):
        return {'ffjord': {'cond': c}}


class TopicFlow(Model):

    # A normalizing flow algorithm for learning subtracted distributions of the form
    # p(x) = p1(x) - kappa * p2(x)

    def __init__(self, dim, kappa, sub_col, flow_arch, flow_config, **kwargs):

        super(TopicFlow, self).__init__(**kwargs)
        self.dim = dim
        self.kappa = tf.Variable(kappa, trainable=False, name='kappa')
        self.sub_col = sub_col
        self.flow = flow_arch(**flow_config)
        self.loss_tracker = metrics.Mean('loss')

    @tf.function
    def forward(self, x, y, kappa, sub_col):
        ll = self(x)
        mask = y[:, sub_col] == 1
        loss = kappa * tf.reduce_mean(ll[mask]) - tf.reduce_mean(ll[~mask]) # Eq. 6
        return loss

    @tf.function
    def call(self, x):
        return self.flow(x)

    @property
    def metrics(self):
        return [self.loss_tracker]

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.forward(x[0], x[1], self.kappa, self.sub_col)
        grads = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.flow.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {self.loss_tracker.name: self.loss_tracker.result()}

    @tf.function
    def test_step(self, x):
        loss = self.forward(x[0], x[1], self.kappa, self.sub_col)
        self.loss_tracker.update_state(loss)
        return {self.loss_tracker.name: self.loss_tracker.result()}

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()


def conv_layer(depth, name, weight_init):
    return tf.keras.layers.Conv2D(filters=depth,
                                  kernel_size=3,
                                  strides=1,
                                  padding="same",
                                  name=name,
                                  kernel_initializer=weight_init)


def residual_block(x, depth, prefix, dropout_prob, weight_init):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_prob)(x)
    x = conv_layer(depth, name=prefix + "_conv0", weight_init=weight_init)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_prob)(x)
    x = conv_layer(depth, name=prefix + "_conv1", weight_init=weight_init)(x)
    return x + inputs


def conv_sequence(x, depth, prefix, dropout_prob=0, weight_init="default"):
    x = conv_layer(depth, prefix + "_conv", weight_init=weight_init)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0", dropout_prob=dropout_prob, weight_init=weight_init)
    x = residual_block(x, depth, prefix=prefix + "_block1", dropout_prob=dropout_prob, weight_init=weight_init)
    return x


class DQNImpalaCNN(DistributionalQTFModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)
        opt = model_config["custom_options"]

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(opt["num_filters"]):
            x = conv_sequence(x,
                              depth,
                              prefix=f"seq{i}",
                              dropout_prob=opt["dropout_prob"],
                              weight_init=opt["weight_init"])

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(opt["dropout_prob"])(x)
        features = tf.keras.layers.Dense(units=num_outputs, name="features", kernel_initializer=opt["weight_init"])(x)
        features = tf.keras.layers.Dropout(opt["dropout_prob"])(features)
        self.base_model = tf.keras.Model(inputs, [features])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        features = self.base_model(obs)
        return features, state


# Register model in ModelCatalog
ModelCatalog.register_custom_model("dqn_impala_cnn_tf", DQNImpalaCNN)

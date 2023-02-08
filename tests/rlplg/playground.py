import copy
import logging
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tf_agents.policies import random_py_policy
from tf_agents.typing.types import NestedArray

from rlplg import metrics, npsci
from rlplg.environments.randomwalk import constants, env
from rlplg.learning.approx import modelspec
from rlplg.learning.approx.evaluation import onpolicy as approx_onpolicy
from rlplg.learning.opt import schedules
from rlplg.learning.tabular.evaluation import onpolicy as tabular_onpolicy


class LinearValueFn(modelspec.ValueFnModel):
    """
    Linear approximation function.
    """

    def __init__(self, initial_weights: NestedArray):
        if not isinstance(initial_weights, np.ndarray):
            initial_weights = np.array(initial_weights, dtype=np.float32)
        self._weights = initial_weights

    def predict(self, inputs: modelspec.ValueFnInputs) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """
        value: float = (
            0.0
            if inputs.is_terminal_state is True
            else self._weights.T * inputs.features
        )
        return value

    def gradients(self, inputs: modelspec.ValueFnInputs) -> NestedArray:
        """
        Computes the gradients of the output
        with respect to the weights.
        """
        return copy.deepcopy(inputs.features)

    def predict_and_gradients(
        self, inputs: modelspec.ValueFnInputs
    ) -> Tuple[float, NestedArray]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """
        return self.predict(inputs), self.gradients(inputs)

    def weights(self) -> NestedArray:
        """
        Returns the current weights of the model.
        """
        return copy.deepcopy(self._weights)

    def assign_weights(self, weights: NestedArray) -> None:
        """
        Assigns new values to the model weights.
        """
        self._weights = copy.deepcopy(weights)

    def __repr__(self) -> str:
        return f"{type(self).__name__}"


class TfDeepLearningModel(modelspec.ValueFnModel):
    """
    Linear approximation function.
    """

    def __init__(self, create_model: Callable[[], tf.keras.Model]):
        self.create_model = create_model
        self.model: Optional[tf.keras.Model] = None

    def predict(self, inputs: modelspec.ValueFnInputs) -> float:
        """
        Computes the value for a given context,
        i.e. state or (state, action)
        """
        if self.model is None:
            self.model = self.create_model()
        value: float = (
            0.0
            if inputs.is_terminal_state is True
            # Output is (batch, output_shape)
            # with batch = 1, we squeeze to get the estimated value.
            # Note: predict (__call__) returns an np.ndarray in
            # eager mode.
            else np.squeeze(self.model(inputs.features, training=False))
        )
        return value

    def gradients(self, inputs: modelspec.ValueFnInputs) -> NestedArray:
        """
        Computes the gradients of the output
        with respect to the weights.
        """
        if self.model is None:
            raise RuntimeError("Model is None.")
        with tf.GradientTape() as tape:
            # TODO: memory clean up of tensor?
            inputs_tensor = tf.constant(inputs.features)
            # Note: Operations are recorded if they are executed within
            # this context manager and at least one of their inputs is being "watched".
            # Hence, we don't need to watch inputs_tensor.

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = self.model(inputs_tensor, training=False)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the output.
        # Note: layer.weights returns a list of [(W0, b0), (W1, b1)... weights
        # while model.vairables returns a list of [W0, b0, W1, b1... variables
        gradients = tape.gradient(target=logits, sources=self._trainable_variables())
        # Some layers will have no weights, e.g. batch normalisation.
        # We set dtype=object to support ragged tensors
        # Return numpy arrays instead of tf variables.
        return np.concatenate([grad.numpy().flatten() for grad in gradients])

    def predict_and_gradients(
        self, inputs: modelspec.ValueFnInputs
    ) -> Tuple[float, NestedArray]:
        """
        Computes prediction and gradients of the
        weights for the prediction.
        """
        return self.predict(inputs), self.gradients(inputs)

    def weights(self) -> NestedArray:
        """
        Returns the current weights of the model.
        """
        # Return np.ndarrays instead of tf.Variables
        # as an external API.
        # Wrap list of variables into an np.ndarray.
        # Note: flatten makes a copy of the array, so we don't have to
        return np.concatenate(
            [variable.numpy().flatten() for variable in self._trainable_variables()],
        )

    def assign_weights(self, weights: NestedArray) -> None:
        """
        Assigns new values to the model weights.
        """
        # rebuild weights
        for variable in self._trainable_variables():
            size = np.size(variable)
            variable_weights, weights = (
                weights[0:size],
                weights[size:],
            )
            variable.assign(np.reshape(variable_weights, newshape=variable.shape))

        # for variable, new_weights in zip(self._trainable_variables(), weights):
        # variable.assign(new_weights)

    def __repr__(self) -> str:
        return f"{type(self).__name__}"

    def _trainable_variables(self) -> NestedArray:
        if self.model is None:
            raise RuntimeError("Model is None.")
        # pass ref to variables with trainable weights
        return [
            variable for variable in self.model.variables if variable.trainable is True
        ]


def create_simple_model_fn(input_shape: Sequence[int]):
    def create_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.Dense(4, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(4, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1))
        return model

    return create_model


def main():
    # environment = suite_gym.load("CliffWalking-v0")
    steps = 3
    # environment = env.StateRandomWalk(steps=steps)
    env_spec = env.create_env_spec(steps=steps)
    environment = env_spec.environment
    environment.reset()
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
    )

    def prepoc(observation: NestedArray) -> modelspec.ValueFnInputs:
        """
        Observation is a single number.
        We add another value for the intercept.
        And another to indicate if the state is terminal.
        """
        is_terminal = env.is_finished(observation)
        # TODO: distance to the right and left
        pos = np.array([observation[constants.OBS_KEY_POSITION]])

        # enrich: e.g. how far to the left, how far to the right is the state
        return modelspec.ValueFnInputs(features=pos, is_terminal_state=is_terminal)

    approx_fn = modelspec.ApproxFn(
        # model=LinearValueFn(initial_weights=np.zeros(shape=(1,), dtype=np.float32)),
        model=TfDeepLearningModel(
            create_model=create_simple_model_fn(input_shape=(1,))
        ),
        pre_proc=prepoc,
    )

    approx_results = approx_onpolicy.gradient_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2000,
        estimator=approx_fn,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1,
            schedule=create_exponential_decay_schedule_fn(base=0.5, mu=100),
        ),
    )
    tabular_results = tabular_onpolicy.first_visit_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2000,
        gamma=1.0,
        initial_values=np.zeros(
            shape=(env_spec.env_desc.num_states,), dtype=np.float32
        ),
        state_id_fn=lambda x: npsci.item(x[constants.OBS_KEY_POSITION]),
    )

    value_table_fn = create_value_table_from_linear_fn(steps)
    for episode, ((_, tabular_vtable), (steps, estimator, delta)) in enumerate(
        zip(tabular_results, approx_results)
    ):
        approx_vtable = value_table_fn(estimator)
        rmse = metrics.rmse(actual=tabular_vtable, pred=approx_vtable)
        if episode % 100 == 0:
            logging.info(
                "Steps: %d; estimator: %s; delta: %f, rmse: %f",
                steps,
                estimator,
                delta,
                rmse,
            )

    logging.info("Tabuler vtable: %s", tabular_vtable)
    logging.info("Approx vtable: %s", approx_vtable)


def create_value_table_from_linear_fn(steps: int):
    states = [
        {
            constants.OBS_KEY_POSITION: np.array(step, dtype=np.int64),
            constants.OBS_KEY_STEPS: np.array(steps, dtype=np.int64),
        }
        for step in range(steps)
    ]

    def value_table_fn(estimator: modelspec.ApproxFn) -> NestedArray:
        return np.array(
            [estimator.predict(obs) for obs in states],
            dtype=float,
        )

    return value_table_fn


def create_exponential_decay_schedule_fn(base: float, mu: int):
    """
    Creates a fn to compute a decaying learning rate.
    alpha_{0} = 0.1, $alpha_{episode} = alpha_{0} * {base}^{floor(episode mod mu)}

    Args:
        base: base in the equation above.
        mu: the range of episodes that correspond to one epoch, e.g.
            if mu = 100, lr is updated every 100 episodes.
    """

    def schedule(initial_learning_rate: float, episode: int, step: int):
        """
        Args:
            initial_learning_rate: the starting value for the first episode, `episode=1`.
            episode: current episode, zero-indexed.
            step: current step, zero-indexed.
        """
        del step
        epoch = math.floor(episode // mu)
        return initial_learning_rate * (base**epoch)

    return schedule


if __name__ == "__main__":
    main()

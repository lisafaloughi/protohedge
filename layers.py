# -*- coding: utf-8 -*-
"""
Utility layers
--------------
June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, tf_glorot_value, Int, Float, DIM_DUMMY# NOQA
from collections.abc import Mapping, Sequence # NOQA
import numpy as np
_log = Logger(__file__)

class VariableLayer(tf.keras.layers.Layer):
    """
    A variable layer.
    The variable can be initialized with a specific value, or with the standard Keras glorot initializer.
    """
    
    def __init__(self, init, trainable : bool = True, name : str = None, dtype : tf.DType = dh_dtype ):
        """
        Initializes the variable

        Parameters
        ----------
            init : 
                If a float, a numpy array, or a tensor, then this is the initial value of the variable
                If this is a tuple, a tensorshape, or a numpyshape then this will be the shape of the variable.
            trainable : bool
            name : str
            dtype : dtype
        """        
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )        
        if not isinstance(init, (float, np.ndarray, tf.Tensor)):
            _log.verify( isinstance(init, (tuple, tf.TensorShape)), "'init' must of type float, np.array, tf.Tensor, tuple, or tf.TensorShape. Found type %s", type(init))
            init                 = tf_glorot_value(init)
        self.variable            = tf.Variable( init, trainable=trainable, name=name+"_variable" if not name is None else None, dtype=self.dtype )
        self._available_features = None

    def build( self, shapes : dict ):
        """
        Build the variable layer
        This function ensures 'shapes' contains DIM_DUMMY so it can create returns of sample size
        """
        self._available_features = sorted( [ str(k) for k in shapes if not k == DIM_DUMMY ] )
        dummy_shape = shapes.get(DIM_DUMMY, None)
        _log.verify( not dummy_shape is None, "Every data set must have a member '%s' (see base.DIM_DUMMY) of shape (None,1). Data member not found data: %s", DIM_DUMMY, list(self.available_features) )
        _log.verify( len(dummy_shape) == 2, "Data set member '%s' (see base.DIM_DUMMY) nust be of shape [None,1], not of shape %s", DIM_DUMMY, dummy_shape.as_list() )
        _log.verify( int(dummy_shape[1]) == 1, "Data set member '%s' (see base.DIM_DUMMY) nust be of shape [None,1], not of shape %s", DIM_DUMMY, dummy_shape.as_list() )
        
    def call( self, dummy_data : dict = None, training : bool = False ) -> tf.Tensor:
        """
        Return variable value
        The returned tensor will be of dimension [None,] if self.variable is a float, and otherwise of dimension [None, ...] where '...' refers to the dimension of the variable.        

        The 'dummy_data' dictionary must have an element DIM_DUMMY of dimension (None,).
        """
        dummy = dummy_data[DIM_DUMMY]
        assert len(dummy.shape) == 2, "Internal error: shape %s not (None,)" % str(dummy.shape.as_list())
        x     = tf.zeros_like(dummy[:,0])
        while len(x.shape) <= len(self.variable.shape):
            x = x[:,tf.newaxis,...]
        x = x + self.variable[tf.newaxis,...]
        return x
    
    @property
    def features(self) -> list:
        """ Returns the list of features used """
        return []
    @property
    def available_features(self) -> list:
        """ Returns the list of features avaialble """
        _log.verify( not self._available_features is None, "build() must be called first")
        return self._available_features
    @property
    def nFeatures(self) -> int:
        """ Returns the number of features used """
        return 0
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )

    
class DenseLayer(tf.keras.layers.Layer):
    """
    Core dense Keras layer
    Pretty generic dense layer. Also acts as plain variable if it does not depend on any variables.
    """
    
    def __init__(self, features, nOutput : int, initial_value = None, config : Config = Config(), name : str = None, defaults = Config(), dtype : tf.DType = dh_dtype ):
        """
        Create a simple dense later with nInput nodes and nOuput nodes.
        
        Parameters
        ----------
            features
                Input features. If None, then the layer will become a simple variable with nOutput nodes.
            nOutput : int
                Number of output nodes
            width : int = 20
            depth : int = 3
            activation : str = "relu"
            name : str, optional
                Name of the layer
            dtype : tf.DType, optional
                dtype
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )
        self.nOutput           = int(nOutput)
        def_width              = defaults("width",20, Int>0, help="Network width.")
        def_activation         = defaults("activation","relu", help="Network activation function")
        def_depth              = defaults("depth", 3, Int>0, help="Network depth")
        def_final_activation   = defaults("final_activation","linear", help="Network activation function for the last layer")
        def_zero_model         = defaults("zero_model", False, bool, "Create a model with zero initial value, but randomized initial gradients")
        self.width             = config("width",def_width, Int>0, help="Network width.")
        self.activation        = config("activation",def_activation, help="Network activation function")
        self.depth             = config("depth", def_depth, Int>0, help="Network depth")
        self.final_activation  = config("final_activation",def_final_activation, help="Network activation function for the last layer")
        self.zero_model        = config("zero_model", def_zero_model, bool, "Create a model with zero initial value, but randomized initial gradients")
        self.features          = sorted( set( features ) ) if not features is None else None
        self.nFeatures         = None
        self.model             = None        
        self.initial_value     = None
        self.available_features= None
        
        if not initial_value is None:
            if isinstance(initial_value, np.ndarray):
                _log.verify( initial_value.shape == (nOutput,), "Internal error: initial value shape %s does not match 'nOutput' of %ld", initial_value.shape, nOutput )
                self.initial_value = initial_value
            else:
                self.initial_value = np.full((nOutput,), initial_value)
                
        _log.verify( self.nOutput > 0, "'nOutput' must be positive; found %ld", self.nOutput )
        config.done()

    def build( self, shapes : dict ):
        """ 
        Keras layer builld() function.
        'shapes' must be a dictionary
        """
        assert self.nFeatures is None and self.model is None, ("build() called twice")
        _log.verify( self.features is None or isinstance(shapes, Mapping), "'shapes' must be a dictionary type if 'features' are specified. Found type %s", type(shapes ))
        
        # collect features
        # features can have different dimensions, so we count the total size of the feature vector
        self.nFeatures = 0
        if not self.features is None:
            for feature in self.features:
                _log.verify( feature in shapes, "Unknown feature '%s'. Known features are: %s. List of requested features: %s", feature, list(shapes), list(self.features) )
                fs = shapes[feature]
                assert len(fs) == 2, ("Internal error: all features should have been flattend. Found feature '%s' with shape %s" % (feature, fs))
                self.nFeatures += fs[1]
                
        self.available_features = sorted( [ str(k) for k in shapes if not k == DIM_DUMMY ] )
    
        # build model
        # simple feedforward model as an example
        if self.nFeatures == 0:
            """ Create model without inputs, but which is trainable.
                Same as creating a plain variable, but wrappong it allows us using
                a single self.model
            """
            self.model    = VariableLayer( (self.nOutput,) if self.initial_value is None else self.initial_value, trainable=True, name=self.name+"_variable_layer" if not self.name is None else None, dtype=self.dtype )
        else:
            """ Simple feed forward network with optional recurrent layer """
            inp = tf.keras.layers.Input( shape=(self.nFeatures,), dtype=self.dtype )
            x = inp
            x = tf.keras.layers.Dense( units=self.width,
                                       activation=self.activation,
                                       use_bias=True )(x)
                                               
            for d in range(self.depth-1):
                x = tf.keras.layers.Dense( units=self.width,
                                           activation=self.activation,
                                           use_bias=True )(x)
            x = tf.keras.layers.Dense(     units=self.nOutput,
                                           activation=self.final_activation,
                                           use_bias=True )(x)
            
            
            self.model         = tf.keras.Model( inputs=inp, outputs=x )
            
            if self.zero_model:
                raise NotImplementedError("zero_model")
                """
                cloned = tf.keras.clone_model( self.model, input_tensors=inp )
                assert len(cloned.weights) == len(self.model.weights), "Internal error: cloned model has differnet number of variables?"
                for mvar, cvar in zip( self.model.weights, cloned.weights):
                    cvar.set_weights(mvar.set_weights)
                cloned.trainable = False
                self.model = tf.keras.layers.
                """  
        
    def call( self, data : dict, training : bool = False ) -> tf.Tensor:
        """
        Ask the agent for an action.
    
        Parameters
        ----------
            data : dict
                Contains all available features at this time step.
                This must be a dictionary.
            training : bool, optional
                Whether we are training or not
                
        Returns
        -------
            Tensor with actions. The second dimension of
            the tensor corresponds to self.nInst
    
        """
        _log.verify( self.features is None or isinstance(data, Mapping), "'data' must be a dictionary type. Found type %s", type(data ))
        _log.verify( not self.model is None, "Model has not been buit yet")

        # simple variable --> return as such
        if self.nFeatures == 0:
            return self.model(data, training=training)
        
        # compile concatenated feature tensor
        features = [ data[_] for _ in self.features ]
        features = tf.concat( features, axis=1, name = "features" )      
        assert self.nFeatures == features.shape[1], ("Condig error: number of features should match up. Found %ld and %ld" % ( self.nFeatures, features.shape[1] ) )
        return self.model( features, training=training )
    
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        assert not self.model is None, "build() must be called first"
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )
    

class ProtoLayer(tf.keras.layers.Layer):
    """
    Prototype Layer
    ---------------
    Computes similarity between inputs and fixed prototypes, and outputs an action
    as a weighted sum of per-prototype trainable actions.

    Similar in style to DenseLayer.
    """

    def __init__(self, features, nOutput: int, prototypes: np.ndarray, config: Config = Config(), name: str = None, dtype: tf.DType = dh_dtype):
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype)
        self.nOutput = int(nOutput)
        self.features = sorted(set(features)) if features is not None else None
        self.nFeatures = None
        self.prototypes_np = prototypes.astype(np.float32)
        self.nPrototypes = prototypes.shape[0]
        self.proto_dim = prototypes.shape[1]
        self.config = config
        self.trainable_action_matrix = None
        self.available_features = None
        config.done()

    def build(self, shapes: dict):
        _log.verify(isinstance(shapes, Mapping), "'shapes' must be a dictionary type.")
        self.nFeatures = 0
        if self.features is not None:
            for feature in self.features:
                _log.verify(feature in shapes, "Unknown feature '%s'.", feature)
                shape = shapes[feature]
                assert len(shape) == 2
                self.nFeatures += shape[1]

        self.available_features = sorted([str(k) for k in shapes if k != DIM_DUMMY])

        _log.verify(self.nFeatures == self.proto_dim, "Feature dimension mismatch: input %d vs prototype %d", self.nFeatures, self.proto_dim)

        self.prototypes = tf.constant(self.prototypes_np, dtype=self.dtype)

        self.trainable_action_matrix = self.add_weight(
            shape=(self.nPrototypes, self.nOutput),
            initializer="glorot_uniform",
            trainable=True,
            name=self.name + "_proto_actions" if self.name else "proto_actions"
        )

    def call(self, data: dict, training: bool = False) -> tf.Tensor:
        _log.verify(self.features is not None and isinstance(data, Mapping), "'data' must be a dictionary.")
        features = tf.concat([data[k] for k in self.features], axis=1)
        _log.verify(self.nFeatures == features.shape[1], "Feature shape mismatch: expected %d, got %d", self.nFeatures, features.shape[1])

        features_exp = tf.expand_dims(features, 1)  # [B, 1, D]
        protos_exp = tf.expand_dims(self.prototypes, 0)  # [1, P, D]
        distances = tf.reduce_sum(tf.square(features_exp - protos_exp), axis=2)  # [B, P]
        similarities = tf.nn.softmax(-distances, axis=1)  # [B, P]

        action = tf.matmul(similarities, self.trainable_action_matrix)  # [B, nOutput]
        return action

    @property
    def num_trainable_weights(self) -> int:
        weights = self.trainable_weights
        return np.sum([np.prod(w.get_shape()) for w in weights])

    @property
    def available_features(self) -> list:
        _log.verify(self.available_features is not None, "build() must be called first")
        return self.available_features



# class ProtoPNetLayer(tf.keras.layers.Layer):
#     """
#     Core ProtoPNet-like layer
#     This layer learns a set of prototypes and outputs similarity-based scores
#     that are linearly mapped to the output space.
#     """
#     def __init__(self, features, nOutput: int, n_prototypes: int, config: Config = Config(), name: str = None, dtype: tf.DType = dh_dtype):
#         tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype)
#         self.nOutput = int(nOutput)
#         self._features = sorted(set(features)) if features else None
#         self.nFeatures = None
#         self.n_prototypes = n_prototypes
#         self.temperature = config("temperature", 1.0, Float>0, help="Softmax temperature for prototype similarity")
#         self.config = config
#         self.prototypes = None
#         self.output_weights = None
#         self.available_features = None
#         self._step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

#     def build(self, shapes: dict):
#         _log.verify(self._features is None or isinstance(shapes, Mapping), "'shapes' must be a dictionary type if 'features' are specified. Found type %s", type(shapes))
#         self.nFeatures = 0
#         for feature in self._features:
#             _log.verify(feature in shapes, "Unknown feature '%s'. Known features: %s", feature, list(shapes))
#             fs = shapes[feature]
#             assert len(fs) == 2, ("All features must be flattened. Found feature '%s' with shape %s" % (feature, fs))
#             self.nFeatures += fs[1]
#         self.available_features = sorted([str(k) for k in shapes if k != DIM_DUMMY])

#         self.prototypes = self.add_weight(
#             shape=(self.n_prototypes, self.nFeatures),
#             initializer='glorot_uniform',
#             trainable=True,
#             name="prototypes"
#         )
#         self.output_weights = self.add_weight(
#             shape=(self.n_prototypes, self.nOutput),
#             initializer='glorot_uniform',
#             trainable=True,
#             name="prototype_output_weights"
#         )

#     def call(self, data: dict, training: bool = False) -> tf.Tensor:
#         _log.verify(self._features is None or isinstance(data, Mapping), "'data' must be a dictionary type. Found type %s", type(data))
#         features = [data[_] for _ in self._features]
#         x = tf.concat(features, axis=1)

#         x_exp = tf.expand_dims(x, 1)
#         p_exp = tf.expand_dims(self.prototypes, 0)
#         dists = tf.reduce_sum(tf.square(x_exp - p_exp), axis=-1)

#         similarities = tf.nn.softmax(-dists / self.temperature)

#         self._latest_similarities = similarities

#         # === Print only if this is sample 0 ===
#         if "_dimension_dummy" in data:
#             path_index = tf.cast(data["_dimension_dummy"][0][0], tf.int32)
#             if tf.equal(path_index, 0):
#                 tf.print("\n[Step", self._step_counter, "] Similarities (sample 0):", similarities[0])
#                 self._step_counter.assign_add(1)
#                 if tf.equal(self._step_counter, 20): 
#                     self._step_counter.assign(0)
#                 k = 3
#                 top_k = tf.argsort(similarities, axis=1, direction='DESCENDING')[:, :k]  # (batch_size, k)
#                 top_k_weights = tf.gather(similarities, top_k, batch_dims=1)  # (batch_size, k)
#                 top_k_actions = tf.gather(self.output_weights, top_k)  # (batch_size, k, output_dim)

#                 # normalize weights
#                 top_k_weights_norm = top_k_weights / tf.reduce_sum(top_k_weights, axis=1, keepdims=True)

#                 tf.print("Top 3 prototypes (sample 0):", top_k[0])
#                 tf.print("Their similarities:", top_k_weights[0])
#                 tf.print("Their similarities after normalization:", top_k_weights_norm[0])

#                 for i in range(k):
#                     proto_id = top_k[0, i]
#                     action = top_k_actions[0, i]
#                     tf.print(" - Prototype #", proto_id, ": action =", action)

#                 # Show final action for sample 0
#                 final_action = tf.matmul(similarities, self.output_weights)[0]
#                 tf.print("Final Actions =", final_action)

#         # === Always return output for batch ===
#         out = tf.matmul(similarities, self.output_weights)
#         return out

#     @property
#     def num_trainable_weights(self) -> int:
#         weights = self.trainable_weights
#         return np.sum([np.prod(w.get_shape()) for w in weights])

#     @property
#     def features(self) -> list:
#         return self._features
    
#     @property
#     def latest_similarities(self):
#         return self._latest_similarities

class ClusteredProtoLayer(tf.keras.layers.Layer):
    def __init__(self, nInst, prototypes, name=None):
        super().__init__(name=name)
        self.nInst = nInst
        self.prototypes = tf.convert_to_tensor(prototypes, dtype=tf.float32)

        # One action per prototype: shape (n_prototypes, nInst)
        self.prototype_actions = self.add_weight(
            shape=(self.prototypes.shape[0], self.nInst),
            initializer='random_normal',
            trainable=True,
            name='proto_actions'
        )

    def call(self, x):  # x shape: [batch, proto_dim]

        # tf.print("x sample:", x[:3], summarize=-1)
        
        x_exp = tf.expand_dims(x, axis=1)                    # [B, 1, D]
        p_exp = tf.expand_dims(self.prototypes, axis=0)      # [1, P, D]
        distances = tf.reduce_sum(tf.square(x_exp - p_exp), axis=2)  # [B, P]

        similarities = tf.nn.softmax(-distances, axis=1)     # [B, P]

        # tf.print("Distances:", distances[0], summarize=10)
        # dist_to_218 = distances[:, 218]
        # tf.print("Distance to prototype 218:", dist_to_218[:10])
        # tf.print("Softmax similarities:", similarities[0], summarize=-1)

        actions = tf.matmul(similarities, self.prototype_actions)  # [B, action_dim]

        # closest = tf.argmax(similarities, axis=1)
        # tf.print("Closest prototypes:", closest[:10])
        # tf.print("Final action:", actions[0], summarize=-1)
        # tf.print("Max similarity:", tf.reduce_max(similarities[0]))
        return actions
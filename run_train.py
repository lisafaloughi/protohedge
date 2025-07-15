# Adjust import path for SageMaker or project folder structure
import os
import sys
import gc
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pickle import load
import random

# Add project root to path
p = os.getcwd()
dhn = "/deephedging/"
i = p.find(dhn)
if i != -1:
    my_path = p[:i]
    sys.path.append(my_path)
    print("SageMaker: added python path %s" % my_path)
else:
    print(sys.path)

# Imports from Deep Hedging
from deephedging.gym import VanillaDeepHedgingGym
from deephedging.trainer import train
from deephedging.world import SimpleWorld_Spot_ATM
from deephedging.base import npCast
from cdxbasics.config import Config

def train_vanilla(world, val_world):
    print("Deep Hedging AI says hello  ... ", end='')
    config = Config()
    config.world.samples = 10000
    config.world.steps = 20
    config.world.black_scholes = False

    config.gym.objective.utility = "cvar"
    config.gym.objective.lmbda = 1.
    config.gym.agent.network.depth = 3
    config.gym.agent.network.activation = "softplus"
    config.gym.agent.init_delta.active = False

    config.trainer.train.optimizer.name = "adam"
    config.trainer.train.optimizer.learning_rate = 0.001
    config.trainer.train.optimizer.clipvalue = 1.
    config.trainer.train.optimizer.global_clipnorm = 1.
    config.trainer.train.batch_size = None
    config.trainer.train.epochs = 800
    config.trainer.caching.mode = "on"
    config.trainer.train.run_eagerly = None
    config.trainer.visual.epoch_refresh = 5
    config.trainer.visual.confidence_pcnt_lo = 0.25
    config.trainer.visual.confidence_pcnt_hi = 0.75

    world = SimpleWorld_Spot_ATM(config.world)
    val_world = world.clone(samples=world.nSamples // 10)

    tf.debugging.enable_check_numerics()
    gym = VanillaDeepHedgingGym(config.gym)

    train(gym=gym, world=world, val_world=val_world, config=config.trainer)
    r = gym(world.tf_data)
    print("Keys of the dictionary returned by the gym: ", r.keys())

    print("=========================================")
    print("Config usage report")
    print("=========================================")
    print(config.usage_report())
    config.done()

    return gym, r, world

def train_protoHedge(seed, n_prototypes):

    # 1. Load configuration
    config = Config()

    # 2. World setup
    config.world.samples = 10000
    config.world.steps = 20
    config.world.black_scholes = False

    # 3. Gym setup with ProtoAgent
    config.gym.agent.agent_type = "protopnet"  # Tells AgentFactory to use ProtoAgent
    config.gym.agent.features = ["price", "delta", "time_left"]  # Must match prototype dimensions
    config.gym.agent.prototype_path = f"./prototypes_storage/prototypes_stochastic_{n_prototypes}.pkl"

    config.gym.objective.utility = "cvar"
    config.gym.objective.lmbda = 1.

    # 4. Load saved prototypes (used inside ProtoPNetLayer)
    with open(f"./prototypes_storage/prototypes_stochastic_{n_prototypes}.pkl", "rb") as f:
        prototypes = pickle.load(f)

    # 5. Trainer config
    config.trainer.train.epochs = 800
    config.trainer.caching.directory = f"./.deephedging_cache/proto_stoch_{n_prototypes}_trained"

    config.trainer.caching.mode = "on"  # Important to avoid reusing old cache
    config.trainer.visual.epoch_refresh = 5

    # trainer
    config.trainer.train.optimizer.name = "adam"
    config.trainer.train.optimizer.learning_rate = 0.001
    config.trainer.train.optimizer.clipvalue = 1.
    config.trainer.train.optimizer.global_clipnorm = 1.
    config.trainer.train.batch_size = None
    config.trainer.visual.confidence_pcnt_lo = 0.25
    config.trainer.visual.confidence_pcnt_hi = 0.75

    # 6. Build world & val_world
    world = SimpleWorld_Spot_ATM(config.world)
    val_world = world.clone(samples=world.nSamples // 10)

    # 7. Create the gym (automatically uses ProtoAgent via AgentFactory)
    gym = VanillaDeepHedgingGym(config.gym)

    # 8. Train the model using the prototype-based agent
    train(gym=gym, world=world, val_world=val_world, config=config.trainer)

    r = gym(world.tf_data)
    print("Keys of the dictionary returned by the gym: ", r.keys())

    print("=========================================")
    print("Config usage report")
    print("=========================================")
    print( config.usage_report() )
    config.done()

    return gym, r, world

def compare_utilities(test_world, proto_result, vanilla_result):
    utility = npCast(proto_result["utility"])
    utility0 = npCast(proto_result["utility0"])
    utility_v = npCast(vanilla_result["utility"])
    P = test_world.sample_weights

    u0 = np.sum(P * utility0) / np.sum(P)
    u_proto = np.sum(P * utility) / np.sum(P)
    u_vanilla = np.sum(P * utility_v) / np.sum(P)

    plt.figure(figsize=(5, 4))
    plt.bar(["Unhedged", "Vanilla DH", "Proto DH"], [u0, u_vanilla, u_proto], color=["blue", "gray", "green"])
    plt.ylabel("Expected Utility")
    plt.title("Expected Utility Comparison (CVaR@50%)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_prototypes = 1000  # Number of prototypes to use
    vanilla_gym, vanilla_result, vanilla_world = train_vanilla()
    proto_gym, proto_result, proto_world = train_protoHedge(seed=1, n_prototypes=n_prototypes)
    compare_utilities(proto_world, proto_result, vanilla_gym(proto_world.tf_data))

    # Set global seeds (recommended for full reproducibility)
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # 1. Select agent and load scaler
    agent = proto_gym.agent
    with open(f"./prototypes_storage/prototypes_stochastic_{n_prototypes}.pkl", "rb") as f:
        data = load(f)
    scaler = data["scaler"]

    # 2. Generate 1 new reproducible test path
    test_world = proto_world.clone(samples=5000, seed=seed)
    test_result = proto_gym(test_world.tf_data)

    path_num = -100

    # 3. Extract per-step inputs (shape: [20, 2] for price and delta)
    price = test_world.data.features.per_step['price'][path_num]          # [20, 2]
    time_left = test_world.data.features.per_step['time_left'][path_num]  # [20,]
    actions = test_result['actions'][path_num]                            # [20, 2]
    delta = np.cumsum(actions, axis=0) - actions                   # [20, 2]

    # 4. Build input: [delta_1, delta_2, price_1, price_2, time_left]
    delta_1 = delta[:, 0]
    delta_2 = delta[:, 1]
    price_1 = price[:, 0]
    price_2 = price[:, 1]
    X = np.stack([delta_1, delta_2, price_1, price_2, time_left], axis=1)   # shape (20, 5)

    # Use actual data extracted from your notebook
    steps = np.arange(20)

    # Replace these with the actual extracted values from path -10
    # These should already be defined:
    # delta_1, delta_2, price_1, price_2

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left Y-axis: Delta
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Delta', color='tab:blue')
    ax1.plot(steps, delta_1, label='Underlying Asset Delta', color='tab:blue', linewidth=2)
    ax1.plot(steps, delta_2, label='ATM Option Delta', color='tab:cyan', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Right Y-axis: Price
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price', color='tab:red')
    ax2.plot(steps, price_1, label='Spot Price', color='tab:red', linestyle='--')
    # ax2.plot(steps, price_2, label='Price 2', color='tab:pink', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    # plt.title('Delta and Price Evolution Over Time (Single Path)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print top 3 prototypes and their actions
    X_scaled = scaler.transform(X)

    # Get model components
    proto_layer = agent.proto_layer
    prototypes = proto_layer.prototypes.numpy()
    proto_actions = proto_layer.prototype_actions.numpy()

    # Convert prototypes and actions to tensors
    feature_weights = tf.convert_to_tensor(proto_layer.feature_weights.numpy(), dtype=tf.float32)  # shape (D,)

    # 8. Compute distances and similarities with weighting
    x_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
    x_exp = tf.expand_dims(x_tf, axis=1)                         # [20, 1, 3]
    p_exp = tf.expand_dims(proto_layer.prototypes, axis=0)       # [1, 500, 3]

    diff = x_exp - p_exp                                               # [T, P, D]
    weighted_diff = diff * feature_weights                             # [T, P, D]
    distances = tf.reduce_sum(tf.square(weighted_diff), axis=2)       # [T, P]
    similarities = tf.nn.softmax(-distances, axis=1)                   # [T, P]
    closest = tf.argmax(similarities, axis=1).numpy()                  # [T]

    # Display step-by-step explanation
    for t in range(X.shape[0]):
        top3_idx = np.argsort(-similarities[t].numpy())[:3]
        print(f"\nStep {t}:")
        print(f"  Input:        delta=({delta[t, 0]:.3f}, {delta[t, 1]:.3f}), price=({price[t, 0]:.3f}, {price[t, 1]:.3f}), time_left={time_left[t]:.3f}")
        print(f"  Action:       ({actions[t, 0]:.4f}, {actions[t, 1]:.4f})")
        print(f"  Top 3 Prototypes:")
        for idx in top3_idx:
            raw_features = scaler.inverse_transform([prototypes[idx]])[0]
            print(f"    • Prototype {idx} | Features={raw_features.round(3)} | Action={proto_actions[idx].round(3)} | Similarity={similarities[t, idx].numpy():.3f}")

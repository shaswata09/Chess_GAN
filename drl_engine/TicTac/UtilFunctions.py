import numpy as np
import pickle
import RLEngine


def create_network(size):
    net_biases_path = "./network/bias.pkl"
    net_weights_path = "./network/weights.pkl"
    net_size_path = "./network/size.pkl"

    print("Starting Neural Network Creation...")
    net = RLEngine.Network(size)
    with open(net_size_path, 'wb') as f:
        pickle.dump(size, f)
    with open(net_biases_path, 'wb') as f:
        pickle.dump(net.biases, f)
    with open(net_weights_path, 'wb') as f:
        pickle.dump(net.weights, f)
    print("Neural Network has been Successfully created and saved to its corresponding files.")


def read_network():
    net_biases_path = "./network/bias.pkl"
    net_weights_path = "./network/weights.pkl"
    net_size_path = "./network/size.pkl"

    print("Starting to read Neural Network files...")
    with open(net_size_path, 'rb') as f:
        size = pickle.load(f)
    net = RLEngine.Network(size)
    with open(net_biases_path, 'rb') as f:
        net.biases = pickle.load(f)
    with open(net_weights_path, 'rb') as f:
        net.weights = pickle.load(f)
    print("Neural Network has been Successfully read from its corresponding files.")
    return net


def update_network(network):
    net_biases_path = "./network/bias.pkl"
    net_weights_path = "./network/weights.pkl"

    print("Starting Neural Network update...")
    with open(net_biases_path, 'wb') as f:
        pickle.dump(network.biases, f)
    with open(net_weights_path, 'wb') as f:
        pickle.dump(network.weights, f)
    print("Neural Network has been Successfully updated and saved to its corresponding files.")



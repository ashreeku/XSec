"""
This file must be run with an environment created using requirements_xnids.txt
"""

import asgl
import json
import pickle
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from timeit import default_timer
from tqdm import trange

from utils import *


class Explanation:
    def __init__(self, current_sample, history_samples, model, original_score, feature_names, group_sizes, target):
        self.current_sample = current_sample
        self.history_samples = history_samples
        self.model = model
        self.original_score = original_score
        self.feature_names = feature_names
        self.group_size = group_sizes
        self.target = target
        self.relevant_history = [] # List to store relevant history samples
        self.delta = 0.001
        self.step = 10
        self.new_input = []
        self.weighted_samples = []
        self.coef = []

    def search_proper_input(self):
      step = 0  # Current step
      for i in range(len(self.history_samples)):
          if step <= self.step:
              self.new_input = self.history_samples[-(i+1):]
              print(len(self.new_input))
              new_current_score = self.model.predict(self.new_input,batch_size=1)
              current_score  = new_current_score [-1]
              print("searched current score", current_score)
              current_delta  = abs(current_score - self.original_score)

              if current_delta <= self.delta:
                return self.new_input  # Found proper input
              step += 1
              i *= 2  # Double the value of i
          else:
            print("Cannot find the proper input within the max steps")
            return self.current_sample  # Cannot find proper in1put

      return None  # Proper input not found within max_steps



    def capture_relevant_history(self):
        """
        Capture the relevant history samples by running the model on current and historical samples.
        """

        if self.target in ['netflow']:
          current_prediction = self.model.predict(self.current_sample)
          current_score = current_prediction

        # Check the difference between current score and the original score
        difference = abs(current_score - self.original_score)
        print("original score:", self.original_score)
        print("difference:",difference)
        if difference > self.delta:
          print("The output is determined by current input and relevant history.")
          # approximate the history inputs here.
          self.new_input = self.history_samples
          self.new_input = self.search_proper_input()
        else:
          print("The output is determined by current input.")
          self.new_input = self.current_sample


    def weighted_sampling(self, num_samples):

        self.weighted_samples = np.array(self.weighted_samples)
        for idx, input_value in enumerate(self.new_input):
            distance = np.abs(idx+1)  # Distance from the current input position
            # Determine the number of selected features based on the distance
            new_weight = distance / (distance +1)
            print("weight:", new_weight)

            for i in range(num_samples):
                if self.target == 'netflow':
                    random_sample39 =np.zeros((1, 39))
                    # Determine the number of selected features based on the distance
                    num_selected = int(new_weight * 39/4)
                    np.random.seed()  # Reset the random seed for each random sample

                    # Select random indices for the features to include in the random sample
                    selected_indices = np.random.choice(39, size=num_selected, replace=False)
                    #print("shape38", random_sample38.shape)
                    #print("shape input", input_value.shape)
                    # Set the selected features to their corresponding values from the input
                    input_value39 = input_value[:, :39]
                    random_sample39 = random_sample39.reshape(1, 39)  # Reshape random_sample38 to match input_value38

                    random_sample39[:, selected_indices] = input_value39[:, selected_indices]
                    #random_sample3 = np.zeros((1, 3))
                    #index = np.random.randint(0, 3)
                    #random_sample3[0, index] = 1
                    #random_sample70 = np.zeros((1, 70))
                    #index = np.random.randint(0, 70)
                    #random_sample70[0, index] = 1
                    #random_sample11 = np.zeros((1, 11))
                    #index = np.random.randint(0, 11)
                    #random_sample11[0, index] = 1
                    #random_sample = np.concatenate((random_sample38, random_sample3,random_sample70,random_sample11), axis=1)
                    random_sample = np.copy(random_sample39)
                # Add the random sample to the list
                self.weighted_samples = np.append(self.weighted_samples, random_sample)

    def sparse_group_lasso(self):
        # Placeholder implementation: Calculate the weights using sparse group lasso
        group_index = []
        for index, value in enumerate(self.group_size):
            group_index.extend([index + 1] * value)
        #print(group_index)
        #y_scores = model.predict(self.weighted_samples.reshape(-1,1,len(self.feature_names)))
        if self.target in ['netflow']:
            y_scores = self.model.predict(self.weighted_samples.reshape(-1,1,len(self.feature_names)))

        #print(y_scores)
        x = self.weighted_samples.reshape(len(y_scores), -1)
        #print(np.shape(x))
        y = y_scores.reshape(len(y_scores))
        #print(y_scores)
        # Define parameters grid
        lambda1 = (10.0 ** np.arange(-3, 1.01, 0.6)).tolist()
        #lambda1 = (0.001, 0.01, 0.1, 1, 10)
        alpha = np.arange(0, 1, 0.2).tolist()
        #alpha = (0, 0.1, 0.2, 0.4)
        power_weight = [0, 0.2, 1]

        # Define model parameters
        model = 'lm'
        penalization = 'sgl'
        tau = 0.5

        # Define cv class
        cross_validation_class = asgl.CV(model=model, penalization=penalization, lambda1=lambda1, alpha=alpha,
                               tau=0.5, parallel=True, weight_technique='pca_pct',
                               lasso_power_weight=power_weight, gl_power_weight=power_weight, variability_pct=0.85,
                               nfolds=5, error_type='QRE', random_state=42)

        # Compute error using k-fold cross validation
        error = cross_validation_class.cross_validation(x, y, group_index)

        # Obtain the mean error across different folds
        error = np.mean(error, axis=1)

        # Select the minimum error
        minimum_error_idx = np.argmin(error)

        # Select the parameters index associated to mininum error values
        optimal_parameters = cross_validation_class.retrieve_parameters_value(minimum_error_idx)

        # Define asgl class using optimal values
        asgl_model = asgl.ASGL(model=model, penalization=penalization, tau=tau,
                       intercept=cross_validation_class.intercept,
                       lambda1=optimal_parameters.get('lambda1'),
                       alpha=optimal_parameters.get('alpha'),
                       lasso_weights=optimal_parameters.get('lasso_weights'),
                       gl_weights=optimal_parameters.get('gl_weights'))

        # Split data into train / test
        train_idx, test_idx = asgl.train_test_split(nrows=x.shape[0], train_pct=0.7, random_state=1)

        # Solve the model
        asgl_model.fit(x=x[train_idx, :], y=y[train_idx], group_index=group_index)

        # Obtain betas
        self.coef = asgl_model.coef_

        #print("coef:", self.coef)
        return self.coef[0][:-1]

    def visualization(self, group_sizes, group_names, feature_names):
        # Get weights and group features
        weights = self.coef[0]
        #print(weights)
        #print(np.shape(weights))
        # Normalize weights to range 0-1
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        #group_features = self.group_features

        # Create color bar visualization
        plt.figure(figsize=(50, 3))
        cmap = plt.colormaps.get_cmap("coolwarm")
        colors = cmap(weights)
        start_index = 0
        total_features = sum(group_sizes)
        feature_index = 0

        # Iterate over groups
        for group_size, group_name in zip(group_sizes, group_names):
            group_weights = weights[start_index:start_index + group_size]
            group_colors = colors[start_index:start_index + group_size]
            group_labels = feature_names[feature_index:feature_index + group_size]

            # Plot color bars for features in the group
            for i, (weight, color, label) in enumerate(zip(group_weights, group_colors, group_labels)):
                plt.bar(feature_index + i, weight, color=color)

                # Connect the feature to its group name with a line
                plt.plot([feature_index + i, feature_index + group_size//2], [weight, 1.3*max(weights)], color='grey', linestyle='-')

            # Add group name below x-axis with adjusted spacing
            plt.text(feature_index + group_size // 2, 1.5, group_name, ha='center', va='top')

            feature_index += group_size
            start_index += group_size

        # Set x-axis ticks and labels
        plt.xticks(range(total_features), feature_names, rotation=45, ha='right')

        # Set y-axis label, x-axis label, and title
        #plt.ylabel("Weights")
        #plt.xlabel("Features")
        #plt.title("Weights Visualization")

        # Remove top, left, and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Hide y-axis ticks
        plt.yticks([])

        # Add color bar legend
        # Get the current axes
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label="Importance Score",shrink=0.5, location='right', pad=0.00, anchor=(0, 0))
        plt.ylim(0, top= 2)  # Set the new minimum and maximum values for the y-axis
        # Display the figure
        plt.show()


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    assert config["dataset"] in ["netflow"], "xNIDS is run only on network datasets"

    set_seed(config["seed"])
    num_seeds = 2
    seed = random.sample(range(1, 10000), num_seeds)[1 if args["alternate"] else 0]
    print(f"Running seed {seed}")
    set_seed(seed)

    x_train, y_train, x_test, y_test = load_data(config["data_path"])
    num_samples, num_features = x_test.shape
    headers = [f"feature_{i}" for i in range(num_features)]

    model = keras.models.load_model(f"../checkpoint/{config['dataset']}/model_xnids.h5")

    if args["train"]:
        x_train = np.expand_dims(x_train, axis=1)
        train_probabilities = model.predict(x_train)
    else:
        x_test = np.expand_dims(x_test, axis=1)
        test_probabilities = model.predict(x_test)

    num_weighted_samples = 100
    group_sizes = [1] * 39
    importance_scores = []
    indices = []
    start_time = default_timer()
    for i in trange(num_samples):
        try:
            current_sample = np.expand_dims(x_test[i], axis=0)
            if args["train"]:
                explanation = Explanation(current_sample, x_train[i], model, train_probabilities[i], headers, group_sizes, config["dataset"])
            else:
                explanation = Explanation(current_sample, x_test[i], model, test_probabilities[i], headers, group_sizes, config["dataset"])
            explanation.capture_relevant_history()
            explanation.weighted_sampling(num_weighted_samples)
            w = torch.from_numpy(explanation.sparse_group_lasso())
            importance_scores.append(w)
            indices.append(i)
        except:
            pass
    
    importance_scores = torch.stack((importance_scores))
    end_time = default_timer()
    print(f"Latency {end_time - start_time} seconds")

    data = {
        "seed": seed,
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/xnids_train_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/xnids_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
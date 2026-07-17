import numpy as np

tokens = ["black","blue","white","grey"]
probabilities = [0.1,0.3,0.5,0.2]
next_token = tokens[np.argmax(probabilities)]
print("Sky Colour is " ,next_token)

#Temperature Sampling
x = np.array([2.0,1.5,1.0])
temperature = 0.5
scaled_x = x/temperature
#Top K Sampling
tokens = ["black","blue","white","grey"]
probabilities = [0.1,0.3,0.5,0.2]
k = 2
top_k_indices = np.argsort(probabilities)[-k:]
top_k_tokens = [tokens[i] for i in top_k_indices]
print("Top K Tokens: ", top_k_tokens)

#Top P Sampling
p = 0.8
probabilities = np.array(probabilities)
cumulative_probs = np.cumsum(probabilities)
top_p_indices = np.where(cumulative_probs <= p)[0]
top_p_tokens = [tokens[i] for i in top_p_indices]
print("Top P Tokens: ", top_p_tokens)
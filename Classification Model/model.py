import torch.nn 


class BreastCancerClassifier(torch.nn.Module):

    def __init__(self, number_of_features = 30):
        super().__init__()

        # Take 30 measurements and produce one raw score.
        # We can add more layers here if I want, just need to use a ReLU
        self.linear = torch.nn.Linear(number_of_features, 1)

        #self.layer1 = torch.nn.Linear(number_of_features, 16)
        #self.layer2 = torch.nn.Linear(16, 8)
        #self.output_layer = torch.nn.Linear(8, 1)
        #self.relu = torch.nn.ReLU()

    def forward(self, features):
        # Returns raw score/logit need sigmoid function later
        raw_scores = self.linear(features)

        #x = self.layer1(features)
        #x = self.relu(x)
        #x = self.layer2(x)
        #x = self.relu(x)
        #raw_scores = self.output_layer(x)
    
        return raw_scores

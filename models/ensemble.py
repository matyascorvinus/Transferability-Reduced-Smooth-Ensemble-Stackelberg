
import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, models, alpha = 1/3, required_alpha=False):
        super(Ensemble, self).__init__()
        self.models = models
        self.alpha = alpha
        self.required_alpha = required_alpha
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            # TODO: Replace the denominator len(self.models) to a alpha number, derived from the BARBE paper

            outputs = 0
            index = 0
            for model in self.models:
                if self.required_alpha:
                    outputs += F.softmax(model(x), dim=-1) * self.alpha[index]
                    index = index + 1
                else:
                    # Original TRS
                    outputs += F.softmax(model(x), dim=-1)
                    index = index + 1
            if self.required_alpha:
                output = outputs
                output = torch.clamp(output, min=1e-40)
                return torch.log(output)
            else:
                output = outputs / len(self.models)
                output = torch.clamp(output, min=1e-40)
                return torch.log(output)
            # output = outputs / len(self.models) 
            # output = torch.clamp(output, min=1e-40)
            # return torch.log(output)
        else:
            return self.models[0](x)
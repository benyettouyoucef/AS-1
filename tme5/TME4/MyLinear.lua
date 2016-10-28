require 'nn'
local MyLinear, parent = torch.class('nn.MyLinear', 'nn.Module')

function MyLinear:reset()
    self.weight = self.weight:uniform(-1, 1)
end

function MyLinear:__init(inputSize, outputSize)
   parent.__init(self)
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self:reset()
end


function MyLinear:updateOutput(input)  
    return self.weight*input
end


function MyLinear:updateGradInput(input, gradOutput) 
    self.gradInput = self.weight:t() * gradOutput
    return self.gradInput
end

function MyLinear:accGradParameters(input, gradOutput)
    self.gradWeight = self.gradWeight+gradOutput*input
    return self.gradWeight
end


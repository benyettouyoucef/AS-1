require 'nn'
require 'torch'

local requ, Parent = torch.class('nn.requ', 'nn.Module')


function requ:__init()
	self.gradInput = torch.Tensor()
	self.output = 0
end 



function requ:updateOutput(input)  
	self.output = torch.cmax(input,0)
	self.output:pow(2)	
	return self.output
end



function requ:updateGradInput(input,gradoutput)
	self.gradInput = input*2
	self.gradInput:cmul(gradoutput)
	return self.gradInput
end





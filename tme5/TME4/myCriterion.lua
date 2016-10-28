require 'nn'

myCriterion, Parent = torch.class('nn.myCriterion', 'nn.Criterion')


function myCriterion:__init(d)
	self.gradInput = torch.Tensor()
	self.output = 0
end 


function myCriterion:forward(input, target) -- appel generique pour calculer le cout
	return self:updateOutput(input, target)
end


function myCriterion:updateOutput(input, target)  
	local n = input:size(1)	
	for i=1,n do	
		diff = input[i]-target[i]
		self.output = self.output + torch.abs(diff)
	end
	return self.output
end

function myCriterion:backward(input, target)  -- appel generique pour calculer le gradient du cout
	return self:updateGradInput(input, target)
end

function myCriterion:updateGradInput(input, target) 
	local n = input:size(1)
	self.gradInput = torch.Tensor(n,1)
	for i=1,n do
		self.gradInput[i] = torch.sign(diff)[1]	
	end
	return self.gradInput
end

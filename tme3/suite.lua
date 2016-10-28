require 'nn'


local MyCriterion, parent = torch.class('MyCriterion', 'nn.Criterion') -- heritage en torch

function MyCriterion:__init() -- constructeur
-- equivalent a  parent.__init(self)
	self.gradInput = torch.Tensor()
	self.output = 0
end

function MyCriterion:forward(input, target) -- appel generique pour calculer le cout
	return self:updateOutput(input, target)
end

function MyCriterion:updateOutput(input, target)
	local seuil = 0.5
	local sum = 0
	for i=1,input:size() do	
		if torch.abs(input[i]-target[i])<= seuil then
			sum = sum + (1\2)*torch.pow((input[i]-target[i]),2)
		else sum = sum + seuil*torch.abs(input[i]-target[i])-(1\2)*torch.pow(seuil,2)
	return sum										-- a completer
end

function Criterion:backward(input, target)  -- appel generique pour calculer le gradient du cout
	return self:updateGradInput(input, target)
end

function MyCriterion:updateGradInput(input, target) -- a completer
	local seuil = O.5	
	local sum = 0	 
	if torch.abs(input[i]-target[i])>= seuil then sum = sum + torch.sign(input-target)*seuil
end




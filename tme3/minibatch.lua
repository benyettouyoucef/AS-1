 require 'nn'
 require 'gnuplot'
 require 'tools' 


local load_mnist = require 'load_mnist'
xtrain,ytrain=load_mnist.get_train(2,3)
xtest,ytest=load_mnist.get_test(2,3)

--xtrain,ytrain,xtest,ytest = load_mnist.get(2,3)
--gnuplot.imagesc(xtrain[1]:reshape(28,28))


 -- 1: Creation du jeux de données
 local DIMENSION=xtrain:size(2) -- dimension d'entrée
 local N=xtrain:size(1) -- nombre de points d'apprentissage
  


function accuracy(y,out)
	local precision = 0
	for i=1, out:size(1)do
		if out[i]*y[i] > 0 then	precision = precision+1
		end
	end
	return precision/out:size(1)
end


 -- 2 : creation du modele
 local model= nn.Linear(DIMENSION, 1)
 local criterion= nn.MSECriterion()
 model:reset(0,1)
 
 
 
 -- 3 : Boucle d'apprentissage
 local learning_rate= 0.01
 local maxEpoch= 500
 local all_losses={}
 local timer = torch.Timer()

 for iteration=1,maxEpoch do
        loss=0 
    	all_losses[iteration]=loss  --stockage de la loss moyenne (pour dessin)
	Blocs = 5
	indexes = torch.randperm(N) --mélanger notre ensemble de données
	x_split = indexes:chunk(Blocs,1) --decouper en Blocs
	y_split = indexes:chunk(Blocs,1)
	for i=1, Blocs do
		model:zeroGradParameters()
		loss = 0
		output = model:forward(xtrain:index(1, x_split[i]:long())) --je fais la descente du gradient par blocs de données
		loss = loss + criterion:forward(output , ytrain:index(1, y_split[i]:long()))
     		grad = criterion:backward(output , ytrain:index(1, y_split[i]:long()))
     		model:backward(xtrain:index(1, x_split[i]:long()) , grad)
     		model:updateParameters(learning_rate)
	end

	all_losses[iteration]=loss/xtrain:size(1)

   --plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
  --plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
  gnuplot.plot(torch.Tensor(all_losses)) 
 end


local function prediction(xtest,ytest)
	output = model:forward(xtest)
	return accuracy(ytest,output)
      end

print(timer:time().real)
print(prediction(xtest,ytest))

--[[
80.883000850677	
0.97600391772772	
]]-- 

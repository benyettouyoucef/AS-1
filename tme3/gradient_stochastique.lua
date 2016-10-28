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
 -- TODO
 local model= nn.Linear(DIMENSION,1)
 model:reset(0.1)
 local criterion= nn.MSECriterion() 
 
 
 
 -- 3 : Boucle d'apprentissage
 local learning_rate= 1e-3 
 local maxEpoch= 500
 local all_losses={}
 local timer = torch.Timer()
 for iteration=1,maxEpoch do
   ------ Evaluation de la loss moyenne 
    -- TODO
    local loss=0  
    local x,y,out,delta
  
    ---- calcul de la loss moyenne 
    for j=1,xtrain:size(1) do
      x=xtrain[j]
      y=ytrain[j]
      out=model:forward(x)
      loss=loss+criterion:forward(out,y)
    end  

    loss=loss/xtrain:size(1)	
    all_losses[iteration]=loss  --stockage de la loss moyenne (pour dessin)
	

    -- version gradient stochastique
    -- TODO    
    model:zeroGradParameters()
    local idx = math.random(xtrain:size(1));
    x=xtrain[idx]
    y=ytrain[idx]      	
    out=model:forward(x)
    loss=criterion:forward(out,y)
    delta=criterion:backward(out,y)
    model:backward(x,delta) 
    model:updateParameters(learning_rate)  

  
    -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
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
85.739324092865	
0.92507345739471		
]]-- 
 

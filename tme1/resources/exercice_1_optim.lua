 require 'nn'
 require 'gnuplot'
 require 'tools' 
 require 'optim'
 
local DIMENSION=2

 -- 1: Creation du jeux de données
 local n_points=100
 local mean_positive=torch.Tensor(DIMENSION):fill(1); local var_positive=1.0
 local mean_negative=torch.Tensor(DIMENSION):fill(-1); local var_negative=1.0
 
 
 local xs=torch.Tensor(n_points,DIMENSION)
 local ys=torch.Tensor(n_points,1)
 
 for i=1,n_points/2 do  xs[i]:copy(torch.randn(DIMENSION)*var_positive+mean_positive); ys[i][1]=1 end
 for i=n_points/2+1,n_points do xs[i]:copy(torch.randn(DIMENSION)*var_negative+mean_negative); ys[i][1]=-1 end
 
 -- 2 : creation du modele
 
 local model=nn.Linear(DIMENSION,1)
 model:reset(0.1)
 local criterion=nn.MSECriterion() 
 params, grad = model:getParameters()
 
 
 
 
 -- 3 : Creatin de la fonction de calcul du gradient
feval = function(params_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if params ~= params_new then
      params:copy(params_new)
   end
   grad:zero()

   -- select a new training sample   
   local idx = math.random(xs:size(1))
   local x=xs[idx]
   local y=ys[idx]
   local out=model:forward(x)
   local loss=criterion:forward(out,y)
   local delta=criterion:backward(out,y)
   model:backward(x,delta)
   
   return loss, grad
end

optim_params = {
   learningRate = 1e-3,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0
}

 local maxEpoch=10000
 local all_losses={}
 
 -- 4 : Boucle d'apprentissage
 
 for iteration=1,maxEpoch do
  ------ Mise à jour des paramètres du modèle
      ------ Evaluation de la loss moyenne 
        ------ Evaluation de la loss moyenne 
    local loss=0
    for j=1,xs:size(1) do
      local x=xs[j]
      local y=ys[j]
      local out=model:forward(x)
      loss=loss+criterion:forward(out,y)
    end
    loss=loss/xs:size(1)
    all_losses[iteration]=loss    
      
    -- apprentissage
    loss=0
    for j=1,xs:size(1) do
      _,fs=optim.sgd(feval,params,optim_params)
      loss=loss+fs[1]
    end
    loss=loss/xs:size(1)  
    
  
  -- plot de la frontiere
 -- plot(xs,ys,model,100)  
  
  -- plot du loss
  gnuplot.plot(torch.Tensor(all_losses))
end
 

 

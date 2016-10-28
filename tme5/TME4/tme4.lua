 require 'torch'
 require 'nn'
 require 'gnuplot'
 require 'tools' 
 require 'requ'
 require 'myCriterion'
 require 'MyLinear'



--Xor


local x=torch.Tensor(200,2)
local y=torch.Tensor(200,1)
for i=1,50 do
        x[i]=torch.randn(2)
        x[i][1]=x[i][1]+3
        x[i][2]=x[i][2]+3
        y[i]=torch.Tensor(1):fill(1)
end
for i=51,100 do
        x[i]=torch.randn(2)
        x[i][1]=x[i][1]-3
        x[i][2]=x[i][2]-3
        y[i]=torch.Tensor(1):fill(1)
end
for i=101,150 do
        x[i]=torch.randn(2)
        x[i][1]=x[i][1]-3
        x[i][2]=x[i][2]+3
        y[i]=torch.Tensor(1):fill(-1)
end
for i=151,200 do
        x[i]=torch.randn(2)
        x[i][1]=x[i][1]+3
        x[i][2]=x[i][2]-3
        y[i]=torch.Tensor(1):fill(-1)
end


-- Affichage des points
--gnuplot.figure()
--gnuplot.plot({x[{{1,100}}],'+' },{x[{{101,200}}],'+'})

--fonction de precision

local function accuracy(y,out)
	local acc = 0
	for i=1, out:size(1)do
		if out[i]*y[i] > 0 then
			acc = acc+1
		end
	end
	return acc/out:size(1)
end






-- 2 : creation du modele
-- TODO
local model1= nn.Linear(2,5)
local model2= nn.Linear(5,1)
--local model1 = nn.MyLinear(2,5)
--local model2 = nn.MyLinear(5,1)


local myCriterion= nn.MSECriterion() 
--local req = nn.ReLU()
local req1 = nn.requ()
local req2 = nn.requ()

local output1=torch.Tensor(5):fill(0) --o1
local y1=torch.Tensor(5):fill(0) --o2
local y2=torch.Tensor(1):fill(0) --o4
local output2=torch.Tensor(1):fill(0) --o3
local delta1 = 0
local delta2 = 0
local delta1req = 0
local delta2req = 0
local idx = 0


-- 3 : Boucle d'apprentissage
local learning_rate= 0.01 
local maxEpoch= 100
local loss = 0
local all_losses={}

for iteration=1,maxEpoch do
	model1:zeroGradParameters()
        model2:zeroGradParameters()
	idx=math.random(x:size(1))
	--forward
	output1 = model1:forward(x[idx])
    	y1 = req1:forward(output1)
	output2 = model2:forward(y1)
   	y2 = req2:forward(output2) 
	loss = loss + myCriterion:forward(y2,y[idx])

	--backward
	delta2 = myCriterion:backward(y2,y[idx])
	delta2req = req2:backward(output2,delta2)
	delta1 = model2:backward(y1,delta2req)
	model2:updateParameters(learning_rate)	
	delta1req = req1:backward(output1,delta1)
	model1:backward(x[idx],delta1req)
	model1:updateParameters(learning_rate)
	
end	


--print(x)
out1 = model1:forward(x)
--print(out1)
out2 = req1:forward(out1)
--print(out2)
out3 = model2:forward(out2)
--print(out3)
pred = req2:forward(out3)
--print(pred)

acc = accuracy(y,pred)

print('Precision : '..acc)



-- utilisation :
--   load_mnist = require 'load_mnist'
--   xtrain,ytrain=load_mnist.get_train(2,3)
--  xtest,test=load_mnist.get_test(2,3)
--  xtrain,ytrain,xtest,ytest = load_mnist.get(2,3)
--  gnuplot.imagesc(xtrain[1]:reshape(28,28))

local mnist = require 'mnist'
local train = mnist.traindataset()
local test = mnist.testdataset()
local full_train = torch.div(torch.reshape(train.data,train.data:size(1),28*28):double(),torch.max(train.data))
local full_test = torch.div(torch.reshape(test.data,test.data:size(1),28*28):double(),torch.max(train.data))
-- gnuplot.imagesc(train.data[1])
--print(train.label[1])


local function get_idx(lab,labels)
  local idx = torch.linspace(1,labels:size(1),labels:size(1)):long()
  return idx[labels:eq(lab)]
end


-- utilisation : par exemple pour recuperer les labels 2 et 5,  get_subsets(2,5,full_train,train.label)

local function get_subsets(labpos,labneg,data,labels)
  local idxpos = get_idx(labpos,labels)
  local idxneg = get_idx(labneg,labels)
  local xset = torch.cat(data:index(1,idxpos),data:index(1,idxneg),1)
  local yset = torch.cat(torch.ones(idxpos:size(1),1),torch.ones(idxneg:size(1),1):fill(-1),1)
  local idx = torch.randperm(yset:size(1)):long()
  return xset:index(1,idx),yset:index(1,idx)
end

local function get_train(labpos,labneg)
  return get_subsets(labpos,labneg,full_train,train.label)
end

local function get_test(labpos,labneg)
  return get_subsets(labpos,labneg,full_test,test.label)
end

local function get(labpos,labneg)
  return get_train(labpos,labneg),get_test(labpos,labneg)
end

return {get_train = get_train, get_test = get_test, get = get}

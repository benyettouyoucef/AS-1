mnist = require 'mnist'
require 'nn';
train = mnist.traindataset()
labels = train.label
reshaped_data = torch.reshape(train.data, 60000,784)

return_value = {}

function mnist_munging(data,labels)
    -- We are going to time the script. It is a good habit to have.
    timer = torch.Timer()    
    times = {}
    times["reshaping"]  = timer:time().real
    -- Reshaping and normalizing
    local reshaped_data = reshaped_data:double()
    reshaped_data = reshaped_data/torch.max(reshaped_data)
    times["reshaping"] = timer:time().real - times["reshaping"]
    print("times : ")
    print(times)
    return reshaped_data
end

return_value["label"] = label
return_value["train"] = train
return_value["reshaped_data"] = reshaped_data
return_value["mnist_munging"] = mnist_munging

function classify_training_examples(reshaped_data,labels)
    -- Now we will gather the training examples by labels.
    timer = torch.Timer()    
    times = {}
    times["classifying"]  = timer:time().real
    -- basic type checking  -- TODO Doesn't work right now.
    if type(labels) == nil then
        return "please provide some good labels"
    end
    
    -- We create the appropriate tensors in order to stock the training examples
    local classified_examples = {}    
    for i=0,9 do
        classified_examples[i] = {}
        classified_examples[i]["data"] = {} 
        classified_examples[i]["count"] = 0
    end
    
    for i=1,(#reshaped_data)[1] do
        classified_examples[labels[i]]["count"] = classified_examples[labels[i]]["count"] + 1
        classified_examples[labels[i]]['data'][classified_examples[labels[i]]["count"]] = reshaped_data[i]
    end
    times["classifying"] = timer:time().real - times["classifying"]
    print("times : ")
    print(times)
    return classified_examples
end

return_value["classify_training_examples"] = classify_training_examples

function convert_to_tensor(data_table)
    -- This method convert the data type from a table to a Tensor
    local result_tensor = torch.Tensor(#data_table,784)
    for i=1,#data_table do
        result_tensor[i] = data_table[i]
    end
    return result_tensor
end

return_value["convert_to_tensor"] = convert_to_tensor
function get_target_classes(classified_dataset,class_1,class_2)
    times = {}
    timer = torch.Timer()
    times["global"] = timer:time().real
    
    --This method returns the targeted classes if they are included into the classes existing in the dataset.
    print("# of example of class ".. class_1 .. " : " .. classified_dataset[class_1].count)
    print("# of example of class " .. class_2 .. " : " .. classified_dataset[class_2].count)
    
    -- We then create a dataset containing all the data with the correct label
    
    local trainset = {}
    
    local look_up_trainset = {}
    -- filling up with class_1 examples 
    for i=1,classified_dataset[class_1].count do 
        look_up_trainset[i] = {}
        look_up_trainset[i]["data"] = classified_dataset[class_1].data[i]
        look_up_trainset[i]["labels"] = class_1
    end
    -- filling up with class_1 examples 
    for i=1,classified_dataset[class_2].count do 
        look_up_trainset[classified_dataset[class_1].count+i] = {}
        look_up_trainset[classified_dataset[class_1].count+i]["data"] = classified_dataset[class_2].data[i]
        look_up_trainset[classified_dataset[class_1].count+i]["labels"] = class_2
    end
    
    times["look_up_building"] = timer:time().real - times["global"]
    times["global"] = timer:time().real
    
    -- We then shuffle the lookup trainset and the labels using the same permutation
    total = classified_dataset[class_1].count + classified_dataset[class_2].count
    -- permutation template
    perm = torch.randperm(total)    
    
    local shuffled_trainset = {}
    shuffled_trainset["data"] = {}
    shuffled_trainset["labels"] = {}
    
    for i=1,total do 
       table.insert(shuffled_trainset["data"],torch.Tensor(look_up_trainset[perm[i]]["data"]))
        if look_up_trainset[perm[i]]["labels"] == class_1 then
            shuffled_trainset["labels"][i] = 1
        else 
            shuffled_trainset["labels"][i] = -1
        end
    end
    shuffled_trainset["labels"] = torch.Tensor(shuffled_trainset["labels"])
    shuffled_trainset["data"] = convert_to_tensor(shuffled_trainset["data"])

    times["shuffling"] = timer:time().real - times["global"]
    -- Adding metatable with __index function allowing heritage from torch
    
    
    setmetatable(shuffled_trainset,
    {__index = function(t, i)
                return {t.data[i], t.labels[i]}
               end})
    function shuffled_trainset:size()
        return self.data:size()
    end
    
    print("times : ")
    print(times)
    return shuffled_trainset
end

return_value["get_target_classes"] = get_target_classes

function mnist_pipeline(class_1,class_2)
    reshaped_data = mnist_munging(mnist.traindataset().data,mnist.traindataset().label)
    classified_data = classify_training_examples(reshaped_data,labels)
    training_set = get_target_classes(classified_data,class_1,class_2)
    return training_set
end

return_value["mnist_pipeline"] = mnist_pipeline

return return_value

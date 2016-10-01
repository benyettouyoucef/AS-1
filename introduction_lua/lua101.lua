local linearModule = {}

local function sayMyName()
  print('Hrunkner')
end

function LinearModule.sayHello()
  print('Why hello there')
  sayMyName()
end

return LinearModule
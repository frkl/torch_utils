--Simple MemNN that runs like LSTM<-Mem. 
--CPU: yes
--Read: yes
--Write: no
--Mem is batch x address x data

require 'nn'
require 'nngraph'
MemNN={};

function MemNN:MemNN(rnn_unit,n,nhaddress,nhdata)
	local init_output=nn.Identity()();
	local init_state=nn.Identity()();
	local data=nn.Identity()();
	local state=init_state;
	local output=init_output;
	local input=nn.View(-1,nhdata)(nn.MM()({nn.View(-1,1,nhaddress)(nn.Normalize(2)(output)),data}));
	for i=1,n-1 do
		local lstm_node=rnn_unit:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var')({state,input});
		state=nn.SelectTable(1)(lstm_node);
		output=nn.SelectTable(2)(lstm_node);
		input=nn.View(-1,nhdata)(nn.MM()({nn.View(-1,1,nhaddress)(nn.Normalize(2)(output)),data}));
	end
	local lstm_node=rnn_unit:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var')({state,input});
	state=nn.SelectTable(1)(lstm_node);
	return nn.gModule({init_state,init_output,data},{state});
end


function MemNN:new(unit,n,nhaddress,nhdata,gpu)
	--unit: {state,input}->{state,output}
	--gpu: yes/no
	gpu=gpu or false;
	local net={};
	local netmeta={};
	netmeta.__index = MemNN
	setmetatable(net,netmeta);
	--stuff
	if gpu then
		net.net=unit:cuda();
	else
	net.net=unit;
	end
	net.w,net.dw=net.net:getParameters();
	net.n=n;
	net.nhaddress=nhaddress;
	net.nhdata=nhdata;
	if gpu then
		net.deploy=self:MemNN(unit,n,nhaddress,nhdata):cuda();
	else
		net.deploy=self:MemNN(unit,n,nhaddress,nhdata)
	end
	collectgarbage();
	net.cell_inputs={};
	return net;
end
function MemNN:training()
	self.net:training();
	self.deploy:training();
end
function MemNN:evaluate()
	self.net:evaluate();
	self.deploy:evaluate();
end
function MemNN:clearState()
	self.net:clearState();
	self.deploy:clearState();
end
function MemNN:forward(a)
	return self.deploy:forward(a);
end
function MemNN:backward(a,b)
	local d=self.deploy:backward(a,b);
	return d;
end
return MemNN
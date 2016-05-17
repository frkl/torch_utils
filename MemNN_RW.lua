--Simple MemNN that runs like LSTM<->Mem. 
--CPU: yes
--Read: yes
--Write: yes
--Mem is batch x address x data

require 'nn'
require 'nngraph'
MemNN_RW={};

function MemNN_RW:MemNN_RW(unit,n)
	local init_memory=nn.Identity();
	local h_prev=nn.Identity();
	local bus_prev=nn.Identity();
	
	local bus=bus_prev;
	local memory=init_memory;
	local h=h_prev;
	
	for i=1,n do
		local node=unit:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var')({h,bus,memory});
		memory=nn.SelectTable(3)(node);
		h=nn.SelectTable(1)(node);
		if i<n then
			bus=nn.SelectTable(2)(node);
		end
	end
	return nn.gModule({h_prev,bus_prev,init_memory},{h,memory});
end

function MemNN_RW:new(cpu_unit,memory_unit,n,nhaddress,nhdata,gpu)
	--unit: {state,input,memory}->{state,memory}
	--gpu: yes/no
	gpu=gpu or false;
	local net={};
	local netmeta={};
	netmeta.__index = MemNN_RW
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
		net.deploy=self:MemNN_RW(unit,n):cuda();
	else
		net.deploy=self:MemNN_RW(unit,n)
	end
	collectgarbage();
	net.cell_inputs={};
	return net;
end
function MemNN_RW:training()
	self.net:training();
	self.deploy:training();
end
function MemNN_RW:evaluate()
	self.net:evaluate();
	self.deploy:evaluate();
end
function MemNN_RW:clearState()
	self.net:clearState();
	self.deploy:clearState();
end
function MemNN_RW:forward(a)
	return self.deploy:forward(a);
end
function MemNN_RW:backward(a,b)
	local d=self.deploy:backward(a,b);
	return d;
end
MemNN_RW.unit={};
function MemNN_RW.unit.MemNN_RW(nhaddress,nhdata,nh,n,dropout)
	local memory=nn.Identity();
	local h_prev=nn.Identity();
	local bus_prev=nn.Identity();
	local new_memory,read_data=MemNN_RW.unit.MemRW_graph(memory,bus_prev,nhaddress,nhdata);
	local h_current,bus_current=MemNN_RW.unit.lstm_graph(h_prev,nhdata,nh,2*nhaddress+2*nhdata,dropout);
	return nn.gModule({h_prev,bus_prev,memory},{h_current,bus_current,new_memory});
end
function MemNN_RW.unit.MemRW_graph(memory,bus,nhaddress,nhdata)
	local write_forget=nn.Sigmoid()(nn.Narrow(2,1,nhdata)(bus));
	local write_address=nn.Normalize(2)(nn.Narrow(2,nhdata+1,nhaddress)(bus));
	local write_data=nn.Narrow(2,nhdata+nhaddress+1,nhdata)(bus);
	local read_address=nn.Normalize(2)(nn.Narrow(2,nhdata*2+nhaddress+1,nhaddress)(bus));
	local old_data=nn.Mul()({nn.View(-1,1,nhaddress)(write_address),memory})
	local diff=nn.CMulTable()({write_forget,nn.CSubTable({old_data,nn.View(-1,1,nhdata)(write_data)})});
	local new_memory=nn.CSubTable()({memory,nn.Mul()({nn.View(-1,nhaddress,1)(write_address),diff})});
	local read_data=nn.Mul()({nn.View(-1,1,nhaddress)(read_address),new_memory});
	return new_memory,read_data;
end

function MemNN_RW.unit.lstm_graph(h_prev,input,nhinput,nh,noutput,n,dropout)
	dropout = dropout or 0 
	local prev_c={};
	local prev_h={};
	for i=1,n do
		prev_c[i]=nn.Narrow(2,2*(i-1)*nh+1,nh)(h_prev);
		prev_h[i]=nn.Narrow(2,(2*i-1)*nh+1,nh)(h_prev);
	end
	local c={};
	local h={};
	local mixed={};
	for i=1,n do
		local x;
		local input_size;
		if i==1 then
			x=input;
			input_size=nhinput;
		else
			x=h[i-1];
			input_size=nh;
		end
		local controls=nn.Linear(input_size+nh,4*nh)(nn.JoinTable(1,1)({x,prev_h[i]}));
		--local sigmoid_chunk=nn.Sigmoid()(nn.Narrow(2,1,3*nh)(controls));
		local data_chunk=nn.Tanh()(nn.Narrow(2,3*nh+1,nh)(controls));
		local in_gate=nn.Sigmoid()(nn.Narrow(2,1,nh)(controls));
		local out_gate=nn.Sigmoid()(nn.Narrow(2,nh+1,nh)(controls));
		local forget_gate=nn.Sigmoid()(nn.Narrow(2,2*nh+1,nh)(controls));
		c[i]=nn.CAddTable()({nn.CMulTable()({forget_gate,prev_c[i]}),nn.CMulTable()({in_gate,data_chunk})});
		h[i]=nn.CMulTable()({out_gate,nn.Tanh()(c[i])});
		table.insert(mixed,c[i]);
		table.insert(mixed,h[i]);
	end
	local h_current=nn.JoinTable(1,1)(mixed);
	local output=nn.Linear(nh,noutput)(nn.Dropout(dropout)(h[n]));
	return h_current,output;
end
return MemNN
--Aggressive variable length RNN
--Designed for word->sentence encoding
require 'nn'
require 'nngraph'
RNN={};
--rnn forward, tries to handle most cases
function RNN:forward(init_state,input,sizes)
	--s0: batch x nh initial state
	--inputs: {batch input for timestep 1} {batch input for timestep 2} ... concatenated in dimension 1. Longer sequence first, left aligned or right aligned, whichever way is faster.
	--sizes: Batch size for each timestep
	--final_state: batch x nh final state
	--outputs: cell array of outputs at each timestep
	
	--first find start and end points for inputs
	local sizes_0=sizes:clone():fill(0);
	sizes_0[{{2,sizes:size(1)}}]=sizes[{{1,sizes:size(1)-1}}];
	local offsets_start=torch.cumsum(sizes_0)+1;
	local offsets_end=torch.cumsum(sizes);
	--loop through timesteps
	local outputs={};
	local final_state=nil;
	local left_aligned=1;
	local state=init_state[{{1,sizes[1]}}];
	for t=1,sizes:size(1) do
		local tmp;
		if t==1 or sizes[t]==sizes[t-1] then
			tmp=self.deploy[t]:forward({state,input[{{offsets_start[t],offsets_end[t]}}]});
		elseif sizes[t]>sizes[t-1] then
			--right aligned
			left_aligned=0;
			if self.cell_inputs[t]==nil then
				self.cell_inputs[t]=init_state[{{1,sizes[t]}}]:clone();
			else
				self.cell_inputs[t]:resizeAs(init_state[{{1,sizes[t]}}]);
				self.cell_inputs[t][{{sizes[t-1]+1,sizes[t]}}]=init_state[{{sizes[t-1]+1,sizes[t]}}];
			end
			self.cell_inputs[t][{{1,sizes[t-1]}}]=state;
			tmp=self.deploy[t]:forward({self.cell_inputs[t],input[{{offsets_start[t],offsets_end[t]}}]});
		elseif sizes[t]<sizes[t-1] then
			--left aligned
			tmp=self.deploy[t]:forward({self.deploy[t-1].output[{{1,sizes[t]}}],input[{{offsets_start[t],offsets_end[t]}}]});
			if final_state==nil then
				final_state=s0:clone();
			end
			final_state[{{sizes[t]+1,sizes[t-1]}}]=self.deploy[t-1].output[{{sizes[t]+1,sizes[t-1]}}];
		end
		state=tmp[1];
		table.insert(outputs,tmp[2]);
	end
	if left_aligned==0 then
		final_state=self.deploy[sizes:size(1)].output[1];
	else
		final_state[{{1,sizes[sizes:size(1)]}}]=self.deploy[sizes:size(1)].output[1];
	end
	return final_state,outputs;
end
--rnn backward
function RNN:backward(init_state,input,sizes,ds,doutputs)
	local sizes_0=sizes:clone():fill(0);
	sizes_0[{{2,sizes:size(1)}}]=sizes[{{1,sizes:size(1)-1}}];
	local offsets_start=torch.cumsum(sizes_0)+1;
	local offsets_end=torch.cumsum(sizes);
	
	if type(doutputs)=="table" then
		--Actually has output gradients
		local N=sizes:size(1);
		local dstate=ds[{{1,sizes[N]}}];
		local dinit_state=nil;
		local dinput_embedding=input:clone():fill(0);
		local left_aligned=1;
		for t=N,1,-1 do
			local tmp;
			if t==1 then
				tmp=self.deploy[t]:backward({init_state[{{1,sizes[1]}}],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,doutputs[t]});
				dstate=tmp[1];
			elseif sizes[t]==sizes[t-1] then
				tmp=self.deploy[t]:backward({self.deploy[t-1].output[1],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,doutputs[t]});
				dstate=tmp[1];
			elseif sizes[t]>sizes[t-1] then
				--right align
				left_aligned=0;
				tmp=self.deploy[t]:backward({self.cell_inputs[t],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,doutputs[t]});
				dstate=tmp[1][{{1,sizes[t-1]}}];
				if dinit_state==nil then
					dinit_state=ds:clone();
				end
				dinit_state[{{sizes[t]+1,sizes[t-1]}}]=tmp[1][{{sizes[t]+1,sizes[t-1]}}];
			elseif sizes[t]<sizes[t-1] then
				--left align
				--compute a larger dstate that matches i-1
				tmp=self.deploy[t]:backward({self.deploy[t-1].output[{{1,sizes[t]}}],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,doutputs[t]});
				dstate_init[{{1,sizes[t]}}]=tmp[1];
				dstate=dstate_init[{{1,sizes[t-1]}}];
			end
			if left_aligned==0 then
				dinit_state[{{1,sizes[1]}}]=self.deploy[1].gradInput;
			else
				dinit_state=self.deploy[1].gradInput;
			end
			dinput_embedding[{{offsets_start[t],offsets_end[t]}}]=tmp[2];
		end
		return dinit_state,dinput_embedding;
	else
		--Actually has output gradients
		local N=sizes:size(1);
		local dstate=ds[{{1,sizes[N]}}];
		local dinit_state=nil;
		local dinput_embedding=input:clone():fill(0);
		local left_aligned=1;
		for t=N,1,-1 do
			local tmp;
			if t==1 then
				tmp=self.deploy[t]:backward({init_state[{{1,sizes[1]}}],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,torch.repeatTensor(doutputs,sizes[t],1)});
				dstate=tmp[1];
			elseif sizes[t]==sizes[t-1] then
				tmp=self.deploy[t]:backward({self.deploy[t-1].output[1],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,torch.repeatTensor(doutputs,sizes[t],1)});
				dstate=tmp[1];
			elseif sizes[t]>sizes[t-1] then
				--right align
				left_aligned=0;
				tmp=self.deploy[t]:backward({self.cell_inputs[t],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,torch.repeatTensor(doutputs,sizes[t],1)});
				dstate=tmp[1][{{1,sizes[t-1]}}];
				if dinit_state==nil then
					dinit_state=ds:clone();
				end
				dinit_state[{{sizes[t-1]+1,sizes[t]}}]=tmp[1][{{sizes[t-1]+1,sizes[t]}}];
			elseif sizes[t]<sizes[t-1] then
				--left align
				--compute a larger dstate that matches i-1
				tmp=self.deploy[t]:backward({self.deploy[t-1].output[{{1,sizes[t]}}],input[{{offsets_start[t],offsets_end[t]}}]},{dstate,torch.repeatTensor(doutputs,sizes[t],1)});
				dstate_init[{{1,sizes[t]}}]=tmp[1];
				dstate=dstate_init[{{1,sizes[t-1]}}];
			end
			dinput_embedding[{{offsets_start[t],offsets_end[t]}}]=tmp[2];
		end
		if left_aligned==0 then
			dinit_state[{{1,sizes[1]}}]=dstate;
		else
			dinit_state=dstate;
		end
		return dinit_state,dinput_embedding;
	end	
end
function RNN:new(unit,n,gpu)
	--unit: {state,input}->{state,output}
	--gpu: yes/no
	gpu=gpu or false;
	local net={};
	local netmeta={};
	netmeta.__index = RNN
	setmetatable(net,netmeta);
	--stuff
	if gpu then
	net.net=unit:cuda();
	else
	net.net=unit;
	end
	net.w,net.dw=net.net:getParameters();
	net.n=n;
	
	net.deploy={};
	for i=1,n do
		net.deploy[i]=net.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	end
	collectgarbage();
	
	net.cell_inputs={};
	return net;
end
function RNN:training()
	self.net:training();
	for i=1,self.n do
		self.deploy[i]:training();
	end
end
function RNN:evaluate()
	self.net:evaluate();
	for i=1,self.n do
		self.deploy[i]:evaluate();
	end
end
function RNN:clearState()
	self.net:clearState();
	for i=1,self.n do
		self.deploy[i]:clearState();
	end
end
RNN.unit={};
function RNN.unit.lstm(nhinput,nh,noutput,n,dropout)
	dropout = dropout or 0 
	local h_prev=nn.Identity()(); -- batch x (2nh), combination of past cell state and hidden state
	local input=nn.Identity()(); -- batch x nhword, input embeddings
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
	return nn.gModule({h_prev,input},{h_current,output});
end

function RNN.unit.gru(nhinput,nh,noutput,n,dropout)
	dropout = dropout or 0 
	local h_prev=nn.Identity()(); -- batch x (2nh), combination of past cell state and hidden state
	local input=nn.Identity()(); -- batch x nhword, input embeddings
	local prev_h={};
	for i=1,n do
		prev_h[i]=nn.Narrow(2,(i-1)*nh+1,nh)(h_prev);
	end
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
		local tx=nn.Linear(input_size,3*nh)(x);
		local th=nn.Linear(nh,2*nh)(prev_h[i]);
		local sigmoid_chunk=nn.Sigmoid()(nn.CAddTable()({nn.Narrow(2,1,2*nh)(tx),th}));
		local output_gate=nn.Narrow(2,1,nh)(sigmoid_chunk);
		local forget_gate=nn.Narrow(2,nh+1,nh)(sigmoid_chunk);
		local data_chunk=nn.Tanh()(nn.CAddTable()({nn.Narrow(2,2*nh+1,nh)(tx),nn.Linear(nh,nh)(nn.CMulTable()({output_gate,prev_h[i]}))}));
		h[i]=nn.CAddTable()({nn.CMulTable()({forget_gate,data_chunk}),nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(forget_gate)),prev_h[i]})});
	end
	local h_current=nn.JoinTable(1,1)(h);
	local output=nn.Linear(nh,noutput)(nn.Dropout(dropout)(h[n]));
	return nn.gModule({h_prev,input},{h_current,output});
end

--GRU from Andrej Karpathy's Char-RNN
function RNN.unit.gru_old(input_size,rnn_size,noutput,n,dropout)
  dropout = dropout or 0 
--my wrapper
	local h_old=nn.Identity()();
	local input=nn.Identity()();
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, input) -- x
  for L = 1,n do
    table.insert(inputs, nn.Narrow(2, (L-1)*rnn_size+1, rnn_size)(h_old)) 
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, noutput)(top_h)
	
	local h_new;
	if n>1 then
		h_new=nn.JoinTable(1,1)(outputs);
	else
		h_new=outputs[1];
	end
	local outs=proj;

  return nn.gModule({h_old,input},{h_new,outs}) 
end

return RNN
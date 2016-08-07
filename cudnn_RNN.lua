--[[
Variable length RNN wrapper for cudnn.LSTM
*  Still does word embeddings in a single pass, but as a length x batch matrix with filler words that have no effect, rather than concatenating slices in word_RNN. 
*  Only works for left-aligned sequences because the filler words will impact the hidden state.
*  The "final_state" is now subselecting outputs. It's no longer concatenation of hidden and cell states.
*  When sentences are shorter than the full RNN length
**     During forward grab only outputs that are relevant
**     During backward slot the doutputs and dstates into the right slots
*  Don't feed in sentences with 0 length!
*  Cudnn only returns the output of the last layer. So the size of the hidden state output is nh.

Usage:
* Initialize a network
            ```
            encoder_net=RNN:new(RNN.unit.lstm(nhword,nh,nhoutput,nlayers,0.5),rnn_length,true);
            ```
			For cudnn, nhoutput must equal to nh, and rnn_length is unused.
* Save a network
            ```
            unit=encoder_net.net;
            torch.save('save.t7',unit);
            ```
* Initialize from checkpoint
            ```
            encoder_net=RNN:new(unit,rnn_length,true);
            ```
			For cudnn rnn_length is unused.
* Forward
            ```
            words_sorted=sort_by_length_left_aligned(words,true);
            word_embeddings=embedding_net:forward(words_sorted.words);
            sentence_embeddings,rnn_outputs=encoder_net:forward(initial_states,word_embeddings,words_sorted.batch_sizes);
            sentence_embeddings=sentence_embeddings:index(1,words_sorted.map_to_sequence);
            ```
			For cudnn initial_states is unused.
* Backward
            ```
            dsentence_embeddings=dsentence_embeddings:index(1,words_sorted.map_to_rnn);
            dinitial_states,dword_embeddings=encoder_net:backward(initial_states,word_embeddings,words_sorted.batch_sizes,dsentence_embeddings,drnn_outputs);
            embedding_net:backward(words_sorted.words,dword_embeddings);
            encoder_net.dw:clamp(-5,5);

            rmsprop(encoder_net.w,encoder_net.dw,opt_encoder);
            rmsprop(embedding_net.w,embedding_net.dw,opt_embedding);
            ```
			For cudnn initial_states is unused.
--]]

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
RNN={};
--Some util functions
--Takes an index (y=x[index]) and return its inverse mapping (x=y[index])
function inverse_mapping(ind)
	local a,b=torch.sort(ind,false);
	return b
end
--Given seq, an N sentence x L length matrix of words, count how many words are in each sentence.
--words are [1..n]
function sequence_length(seq)
	local v=seq:gt(0):long():sum(2):view(-1):long();
	return v;
end

--Join and split Tensors
--Originally https://github.com/torch/nn/blob/master/JoinTable.lua nn.JoinTable:updateOutput()
function join_vector(tensor_table,dimension)
	if dimension==nil then
		dimension=1;
	end
	local size=torch.LongStorage();
	for i=1,#tensor_table do
		local currentOutput = tensor_table[i];
		if i == 1 then
			size:resize(currentOutput:dim()):copy(currentOutput:size());
		else
			size[dimension] = size[dimension] + currentOutput:size(dimension);
		end
	end
	local output=tensor_table[1]:clone();
	output:resize(size);
	local offset = 1;
	for i=1,#tensor_table do
		local currentOutput = tensor_table[i];
		output:narrow(dimension, offset, currentOutput:size(dimension)):copy(currentOutput);
		offset = offset + currentOutput:size(dimension);
	end
	return output;
end
function split_vector(w,sizes,dimension)
	if dimension==nil then
		dimension=1;
	end
	local tensor_table={};
	local offset=1;
	local n;
	if type(sizes)=="table" then
		n=#sizes;
	else
		n=sizes:size(1);
	end
	for i=1,n do
		table.insert(tensor_table,w:narrow(dimension,offset,sizes[i]));
		offset=offset+sizes[i];
	end
	return tensor_table;
end
--Turn a left-aligned N sentence x L length matrix of words to right aligned.
--Allocates new memory
function right_align(seq)
	local lengths=sequence_length(seq);
	local v=seq:clone():fill(0);
	local L=seq:size(2);
	for i=1,seq:size(1) do
		v[i][{{L-lengths[i]+1,L}}]=seq[i][{{1,lengths[i]}}];
	end
	return v;
end


--For cudnn, words is now a seq_length x batch_size matrix rather than a concatenated vector. 
--Batch_sizes is now a [batch_size] vector indicating how long each sentence is. 
--Inputs:  seq: left-aligned sentences
--Outputs: 1) words: words at each time step, concatenated at dimension 1, so they can be fed into a LookupTable in 1 shot for efficiency.
--         2) batch_sizes: batch size at each time step. Starts at batch_size and ends small for left-aligned sentences.
--         3) map_to_rnn: mapping from the sentence order of seq to sentence order of the sorted sentences
--         4) map_to_sequence: mapping from the sorted sentences to the original sentence order of seq.
function sort_by_length_left_aligned(seq,gpu,seq_length)
	gpu=gpu or false;
	seq_length=seq_length or sequence_length(seq);
	local seq_length_sorted,sort_index=torch.sort(seq_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local seq_sorted=seq:index(1,sort_index);
	local L=seq_length_sorted[1];
	if L==0 then
		--We got a batch of empty sentences. There's no words, no batch sizes, so there's nothing to do.
		if gpu then
			return {words=nil,batch_sizes=nil,map_to_rnn=sort_index:cuda(),map_to_sequence=sort_index_inverse:cuda()};
		else
			return {words=nil,batch_sizes=nil,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
		end
	end
	local words=seq_sorted[{{},{1,L}}]:t();
	words[words:eq(0)]=1; --fill the unused words with a garbage word
	local batch_sizes=seq_length_sorted;
	if gpu then
		words=words:cuda();
		sort_index=sort_index:cuda();
		sort_index_inverse=sort_index_inverse:cuda();
	end
	return {words=words,batch_sizes=batch_sizes,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
end


--Variable length RNN forward. Applies to left-aligned sequences, sorted by length in descending order
--Inputs:  1) init_state: not used
--         2) input: sequence_length x batch_size x nhinput
--         3) sizes: a [batch_size] vector indicating how long each sentence is.
--Outputs: 1) final_state: final states (outputs) for all sequences
--         2) outputs: a table of outputs at each time step 
function RNN:forward(init_state,input,sizes)
	if input==nil then
		--Welp, there's no input for whatever reason.
		--For cudnn this would be an error
		error('Input sequence has length 0');
		return init_state,{};
	end
	
	local output=self.deploy:forward(input);
	--output is sequence_length x batch_size x nh
	--Now carefully pick output and final_state for each sentence
	local output_view=output:view(input:size(1)*input:size(2),-1);
	local ind=(sizes-1)*input:size(2)+torch.range(1,input:size(2)):long();
	if self.gpu then
		ind=ind:cuda()
	end
	local final_state=output_view:index(1,ind);
	local outputs={};
	for i=1,input:size(1) do
		table.insert(outputs,output[i][{{1,sizes:ge(i):sum()}}]);
	end
	return final_state,outputs;
end
--Variable length RNN backward. Applies to left-aligned sequences, sorted by length in descending order
--Inputs:  1) init_state: not used
--         2) input: sequence_length x batch_size x nhinput
--         3) sizes: a [batch_size] vector indicating how long each sentence is.
--         4) dfinal_state: gradient to the final state
--         5) doutputs: a table of gradients to the outputs at each time step
--Outputs: 1) dinit_state: final states for all sequences
--         2) dinput: derivative to the inputs, concatenated over time steps.
function RNN:backward(init_state,input,sizes,dfinal_state,doutputs)
	if input==nil then
		--Welp, there's no input for whatever reason.
		--For cudnn this would be an error
		error('Input sequence has length 0');
		return dfinal_state,nil;
	end
	--Then make a tensor for doutputs
	local doutputs_tensor;
	local doutputs_exp;
	if type(doutputs)=='table' then
		--Fill in the tensor with the table
		doutputs_tensor=doutputs[1]:expandAs(self.deploy.output);
		for i=1,#doutputs do
			doutputs_tensor[i][{{1,doutputs[i]:size(1)}}]=doutputs[i];
		end
	else
		--Repeating dummy output gradients
		doutputs_tensor=torch.repeatTensor(doutputs:view(1,1,-1),input:size(1),input:size(2),1);
	end
	--Then add dfinal_state into that tensor. As fast as I possibly can...
	--Approach 1
	for i=1,input:size(1) do
		--The first one that has length le i
		local s=sizes:gt(i):long():sum()+1;
		--The last one that has length le i
		local e=math.min(sizes:ge(i):long():sum(),input:size(2));
		if e>s then
			doutputs_tensor[{i,{s,e}}]=doutputs_tensor[{i,{s,e}}]+dfinal_state[{{s,e}}];
		end
	end
	--Approach 2: much slower indeed
	--for i=1,input:size(2) do
	--	doutputs_tensor[sizes[i]][i]=doutputs_tensor[sizes[i]][i]+dfinal_state[i];
	--end
	local dinput=self.deploy:backward(input,doutputs_tensor);
	return {},dinput;
end

function RNN:new(unit,length,gpu)
	--gpu: yes/no
	gpu=gpu or false;
	local net={};
	local netmeta={};
	netmeta.__index = RNN
	setmetatable(net,netmeta);
	net.net=unit;
	if gpu then
		net.net=net.net:cuda();
	else
		net.net=net.net;
	end
	net.w,net.dw=net.net:getParameters();
	net.n=n;
	net.gpu=gpu;
	net.deploy=net.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	collectgarbage();
	return net;
end
function RNN:training()
	self.net:training();
	self.deploy:training();
end
function RNN:evaluate()
	self.net:evaluate();
	self.deploy:evaluate();
end
function RNN:clearState()
	self.net:clearState();
	self.deploy:clearState();
end

RNN.unit={};
function RNN.unit.lstm(nhinput,nh,noutput,n,dropout)
	if nh~=noutput then
		error('For CUDNN number of hidden units and number of output units must be the same!');
	end
	return cudnn.LSTM(nhinput,nh,n,false);
end

return RNN
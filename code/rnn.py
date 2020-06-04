import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit,grad
import flax
from flax import nn

class RNN2D(nn.Module):
    def apply(self, x, L=10, units=[10], inputDim=2, actFun=nn.elu):
        cellInV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_v',
                                    bias=False)
        cellInH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_h',
                                    bias=False)
        cellCarryV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_v',
                                    bias=False)
        cellCarryH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_h',
                                    bias=True)

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense')

        batchSize = x.shape[0]

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))

        states = jnp.asarray(np.zeros((L,batchSize,units[0])))
        inputs = jnp.asarray(np.zeros((L+1,L+1,batchSize,inputDim)))

        x = jnp.transpose(x,axes=[1,2,0])
        inputs = jax.ops.index_update(inputs,jax.ops.index[1:,1:],jax.nn.one_hot(x,inputDim))
       
        def rnn_dim2(carry,x):
            newCarry = actFun( cellInH(x[0]) + cellInV(x[1]) + cellCarryH(carry) + cellCarryV(x[2]) )
            out = jnp.concatenate((newCarry, nn.softmax(outputDense(newCarry))), axis=1)
            return newCarry, out
        def rnn_dim1(carry,x):
            _, out = jax.lax.scan(rnn_dim2,jnp.zeros((batchSize,units[0]),dtype=np.float64),(x[0][:-1],x[1][1:],carry))
            carry = jax.ops.index_update(carry,jax.ops.index[:,:],out[:,:,:units[0]])
            outputs = jnp.log( jnp.sum( out[:,:,units[0]:] * x[0][1:,:], axis=2 ) )
            return carry, jnp.sum(outputs,axis=0)
        
        _, prob = jax.lax.scan(rnn_dim1,states,(inputs[1:],inputs[:-1]))
        return jnp.sum(prob,axis=0)

#    def apply(self, x, L=10, units=[10], inputDim=2, actFun=nn.elu):
#        cellInV = nn.Dense.shared(features=units[0],
#                                    name='rnn_cell_in_v',
#                                    bias=False)
#        cellInH = nn.Dense.shared(features=units[0],
#                                    name='rnn_cell_in_h',
#                                    bias=False)
#        cellCarryV = nn.Dense.shared(features=units[0],
#                                    name='rnn_cell_carry_v',
#                                    bias=False)
#        cellCarryH = nn.Dense.shared(features=units[0],
#                                    name='rnn_cell_carry_h',
#                                    bias=True)
#
#        outputDense = nn.Dense.shared(features=inputDim,
#                                      name='rnn_output_dense')
#
#        outputs = jnp.asarray(np.zeros((x.shape[0],L,L)))
#
#
#        #states = jnp.asarray(np.zeros((L+1,L+1,x.shape[0],units[0])))
#        #inputs = jnp.asarray(np.zeros((L+1,L+1,x.shape[0],inputDim)))
#
#        states = {}
#        inputs = {}
#        for l in range(L):
#            states[str(-1)+str(l)] = np.zeros((x.shape[0],units[0]))
#            states[str(l)+str(-1)] = np.zeros((x.shape[0],units[0]))
#            inputs[str(-1)+str(l)] = np.zeros((x.shape[0],inputDim))
#            inputs[str(l)+str(-1)] = np.zeros((x.shape[0],inputDim))
#
#        #x = jnp.transpose(x,axes=[1,2,0])
#        #inputs = jax.ops.index_update(inputs,jax.ops.index[:-1,:-1],jax.nn.one_hot(x,inputDim))
#
#        for lx in range(L):
#            for ly in range(L):
#                inputs[self.idx(lx,ly)] = jax.nn.one_hot(x[:,lx,ly],inputDim)
#
#        
#        #for lx in range(L):
#        #    for ly in range(L):
#        #        states = jax.ops.index_update(states,jax.ops.index[lx,ly,:,:],
#        #                       actFun ( cellInH(inputs[lx-1,ly,:,:]) \
#        #                       + cellInV(inputs[lx,ly-1,:,:]) \
#        #                       + cellCarryH(states[lx-1,ly,:,:]) \
#        #                       + cellCarryV(states[lx,ly-1,:,:]) )
#        #                 )
#        #        outputs=jax.ops.index_update(outputs,jax.ops.index[:,lx,ly],
#        #                             jnp.log( jnp.sum( nn.softmax(outputDense(states[lx,ly,:,:])) * inputs[lx,ly,:,:], axis=-1 ) )
#        #                            )
#        
#        for lx in range(L):
#            for ly in range(L):
#                states[self.idx(lx,ly)] = actFun(  cellInH(inputs[self.idx(lx-1,ly)]) \
#                                                   + cellInV(inputs[self.idx(lx,ly-1)]) \
#                                                   + cellCarryH(states[self.idx(lx-1,ly)]) \
#                                                   + cellCarryV(states[self.idx(lx,ly-1)]) \
#                                                  )
#                #outputs[lx,ly] = jnp.log( jnp.dot( nn.softmax(outputDense(states[self.idx(lx,ly)])), inputs[self.idx(lx,ly)] ) )
#                #outputs=jax.ops.index_update(outputs,jax.ops.index[lx,ly],
#                #                     jnp.log( jnp.dot( nn.softmax(outputDense(states[self.idx(lx,ly)])), inputs[self.idx(lx,ly)] ) )
#                #                    )
#                outputs=jax.ops.index_update(outputs,jax.ops.index[:,lx,ly],
#                                     jnp.log( jnp.sum( nn.softmax(outputDense(states[self.idx(lx,ly)])) * inputs[self.idx(lx,ly)], axis=1 ) )
#                                    )
#
#        return jnp.sum(jnp.sum(outputs,axis=2), axis=1)

    def idx(self,lx,ly):
        return str(lx)+str(ly)

    @nn.module_method
    def sample(self,out,key,L,units,inputDim=2,actFun=nn.elu):
        cellInV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_v',
                                    bias=False)
        cellInH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_h',
                                    bias=False)
        cellCarryV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_v',
                                    bias=False)
        cellCarryH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_h',
                                    bias=True)

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense')

        outputs = jnp.asarray(np.zeros((out.shape[0],L,L)))

        dummyState = jnp.asarray([0,1])

        states = {}
        inputs = {}
        for l in range(L):
            states[str(-1)+str(l)] = np.zeros((out.shape[0],units[0]))
            states[str(l)+str(-1)] = np.zeros((out.shape[0],units[0]))
            inputs[str(-1)+str(l)] = np.zeros((out.shape[0],inputDim))
            inputs[str(l)+str(-1)] = np.zeros((out.shape[0],inputDim))
        
        for lx in range(L):
            for ly in range(L):
                states[self.idx(lx,ly)] = actFun(  cellInH(inputs[self.idx(lx-1,ly)]) \
                                                   + cellInV(inputs[self.idx(lx,ly-1)]) \
                                                   + cellCarryH(states[self.idx(lx-1,ly)]) \
                                                   + cellCarryV(states[self.idx(lx,ly-1)]) \
                                                  )
                probabilities=jnp.sum( nn.softmax(outputDense(states[self.idx(lx,ly)])) * dummyState, axis=1 )

                currentKey, key = jax.random.split(key)
                out = jax.ops.index_update( out,jax.ops.index[:probabilities.shape[0],lx,ly],jax.random.bernoulli(currentKey, p=probabilities) )
                #out[:,lx,ly] = jax.random.bernoulli(currentKey, p=probabilities)
                inputs[self.idx(lx,ly)] = jax.nn.one_hot(out[:,lx,ly],inputDim)
                
                outputs=jax.ops.index_update(outputs,jax.ops.index[:,lx,ly],
                                     jnp.log( jnp.sum( nn.softmax(outputDense(states[self.idx(lx,ly)])) * inputs[self.idx(lx,ly)], axis=1 ) )
                                    )

        return out,jnp.sum(jnp.sum(outputs,axis=2), axis=1)

@jit
def get_states():
    stateList=[]
    for j in range(2**9):
        state=jnp.zeros(9)
        for k in range(9):
            if j>>k & 1:
                state=jax.ops.index_update(state,jax.ops.index[k],1)
        stateList.append(jnp.reshape(state,(3,3)))

    return jnp.array(stateList)

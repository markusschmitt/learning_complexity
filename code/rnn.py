import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import jit,grad
import flax
from flax import nn

class RNN2D(nn.Module):
    def apply(self, x, L=10, units=[10], inputDim=2, actFun=nn.elu, initScale=1.0):

        initFunctionCell = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")
        initFunctionOut = jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform")
        #initFunction = jax.nn.initializers.lecun_uniform()

        cellInV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_v',
                                    bias=False)
        cellInH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_h',
                                    bias=False)
        cellCarryV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_v',
                                    bias=False,
                                    kernel_init=initFunctionCell)
        cellCarryH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_h',
                                    bias=True,
                                    kernel_init=initFunctionCell)

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense',
                                      kernel_init=initFunctionOut)

        batchSize = x.shape[0]

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))

        states = jnp.asarray(np.zeros((L,batchSize,units[0])))
        inputs = jnp.asarray(np.zeros((L+1,L+2,batchSize,inputDim)))

        # Scan directions for zigzag path
        direction = np.ones(L,dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)

        x = jnp.transpose(x,axes=[1,2,0])
        inputs = jax.ops.index_update(inputs,jax.ops.index[1:,1:-1],jax.nn.one_hot(x,inputDim))
      
        def rnn_dim2(carry,x):
            newCarry = actFun( cellInH(x[0]) + cellInV(x[1]) + cellCarryH(carry) + cellCarryV(x[2]) )
            out = jnp.concatenate((newCarry, nn.softmax(outputDense(newCarry))), axis=1)
            return newCarry, out
        def rnn_dim1(carry,x):
            _, out = jax.lax.scan(rnn_dim2,jnp.zeros((batchSize,units[0]),dtype=np.float32),
                                    (self.reverse_line(x[0],x[2])[:-2],
                                     self.reverse_line(x[1],x[2])[1:-1],
                                     self.reverse_line(carry,x[2]))
                                 )
            carry = jax.ops.index_update(carry,jax.ops.index[:,:],out[:,:,:units[0]])
            outputs = jnp.log( jnp.sum( out[:,:,units[0]:] * self.reverse_line(x[0],x[2])[1:-1,:], axis=2 ) )
            return self.reverse_line(carry,x[2]), jnp.sum(outputs,axis=0)
        
        _, prob = jax.lax.scan(rnn_dim1,states,(inputs[1:],inputs[:-1],direction))
        return jnp.nan_to_num(jnp.sum(prob,axis=0))

    def reverse_line(self, line, b):
        return jax.lax.cond(b==1, lambda z : z, lambda z : jnp.flip(z,0), line) 


    @nn.module_method
    def sample(self,batchSize,key,L,units,inputDim=2,actFun=nn.elu, initScale=1.0):

        initFunctionCell = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")
        initFunctionOut = jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform")

        cellInV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_v',
                                    bias=False)
        cellInH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_h',
                                    bias=False)
        cellCarryV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_v',
                                    bias=False,
                                    kernel_init=initFunctionCell)
        cellCarryH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_h',
                                    bias=True,
                                    kernel_init=initFunctionCell)

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense',
                                      kernel_init=initFunctionOut)


        outputs = jnp.asarray(np.zeros((batchSize,L,L)))
        
        # Scan directions for zigzag path
        direction = np.ones(L,dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)

        def rnn_dim2(carry,x):
            newCarry = actFun( cellInH(carry[1]) + cellInV(x[1]) + cellCarryH(carry[0]) + cellCarryV(x[0]) )
            logits=outputDense(newCarry)
            sampleOut=jax.random.categorical(x[2],logits)
            sample=jax.nn.one_hot(sampleOut,inputDim)
            logProb=jnp.log( jnp.sum( nn.softmax(logits) * sample, axis=1 ) )
            output = (newCarry, logProb, sampleOut)
            return (newCarry,sample), output
        def rnn_dim1(carry,x):
            keys = jax.random.split(x[0],L)
            _, output = jax.lax.scan(rnn_dim2,(jnp.zeros((batchSize,units[0]),dtype=np.float32),
                                               jnp.zeros((batchSize,inputDim),dtype=np.float32)),
                                     (self.reverse_line(carry[0],x[1]),
                                      self.reverse_line(carry[1],x[1]), keys))
            return (self.reverse_line(output[0],x[1]), self.reverse_line(jax.nn.one_hot(output[2],inputDim), x[1])),\
                    (jnp.sum(output[1], axis=0), self.reverse_line(output[2],x[1]))
        
        keys = jax.random.split(key,L)
        _, res = jax.lax.scan(rnn_dim1,(jnp.zeros((L,batchSize,units[0]),dtype=np.float32),jnp.zeros((L,batchSize,inputDim),dtype=np.float32)),(keys,direction))

        return jnp.transpose(res[1],axes=[2,0,1]), jnp.nan_to_num(jnp.sum(res[0], axis=0))
    
    
    @nn.module_method
    def prob_factors(self, x, L=10, units=[10], inputDim=2, actFun=nn.elu, initScale=1.0):

        initFunctionCell = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")
        initFunctionOut = jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform")
        #initFunction = jax.nn.initializers.lecun_uniform()

        cellInV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_v',
                                    bias=False)
        cellInH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in_h',
                                    bias=False)
        cellCarryV = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_v',
                                    bias=False,
                                    kernel_init=initFunctionCell)
        cellCarryH = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry_h',
                                    bias=True,
                                    kernel_init=initFunctionCell)

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense',
                                      kernel_init=initFunctionOut)

        batchSize = x.shape[0]

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))

        states = jnp.asarray(np.zeros((L,batchSize,units[0])))
        inputs = jnp.asarray(np.zeros((L+1,L+2,batchSize,inputDim)))

        # Scan directions for zigzag path
        direction = np.ones(L,dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)

        x = jnp.transpose(x,axes=[1,2,0])
        inputs = jax.ops.index_update(inputs,jax.ops.index[1:,1:-1],jax.nn.one_hot(x,inputDim))
      
        def rnn_dim2(carry,x):
            newCarry = actFun( cellInH(x[0]) + cellInV(x[1]) + cellCarryH(carry) + cellCarryV(x[2]) )
            out = jnp.concatenate((newCarry, nn.softmax(outputDense(newCarry))), axis=1)
            return newCarry, out
        def rnn_dim1(carry,x):
            _, out = jax.lax.scan(rnn_dim2,jnp.zeros((batchSize,units[0]),dtype=np.float32),
                                    (self.reverse_line(x[0],x[2])[:-2],
                                     self.reverse_line(x[1],x[2])[1:-1],
                                     self.reverse_line(carry,x[2]))
                                 )
            carry = jax.ops.index_update(carry,jax.ops.index[:,:],out[:,:,:units[0]])
            outputs = jnp.log( jnp.sum( out[:,:,units[0]:] * self.reverse_line(x[0],x[2])[1:-1,:], axis=2 ) )
            #outputs = jnp.log( out[:,:,units[0]] )
            return self.reverse_line(carry,x[2]), outputs
        
        _, prob = jax.lax.scan(rnn_dim1,states,(inputs[1:],inputs[:-1],direction))
        return jnp.exp(prob)

def get_states_f(L=3):
    stateList=[]
    N=L*L
    for j in range(2**N):
        state=jnp.zeros(N)
        for k in range(N):
            if j>>k & 1:
                state=jax.ops.index_update(state,jax.ops.index[k],1)
        stateList.append(jnp.reshape(state,(L,L)))

    return jnp.array(stateList)
get_states=jax.jit(get_states_f,static_argnums=0)

if __name__ == "__main__":
    L=3
    s=get_states(L)
    
    # Model setup
    rnn = RNN2D.partial(L=3,units=[20])
    _,params = rnn.init_by_shape(random.PRNGKey(0),[(1,3,3)])
    rnnModel = nn.Model(rnn,params)

    logProb=rnnModel(s)
    print("*** Test normalization")
    nrm=jnp.sum(jnp.exp(logProb))
    if np.abs(nrm-1.)<1e-4:
        ok="=)"
    else:
        ok=":("
    print("    Norm =",nrm,ok)

    print(rnnModel.prob_factors(s))

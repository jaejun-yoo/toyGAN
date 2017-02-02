# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 00:00:31 2017

@author: Jaejun Yoo
"""

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Generative Adversarial Nets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns # for pretty plots
from scipy.stats import norm
tf.reset_default_graph()



# MLP - G networks
def generator(input, output_dim):
    # construct learnable parameters within local scope
    w1=tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, output_dim], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    return fc2, [w1,b1,w2,b2]

# MLP - used for D_pre, D1, D2
def mlp(input, output_dim):
    # construct learnable parameters within local scope
    w1=tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [5,5], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [5], initializer=tf.constant_initializer(0.0))
    w4=tf.get_variable("w3", [5,output_dim], initializer=tf.random_normal_initializer())
    b4=tf.get_variable("b3", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
    fc4=tf.nn.tanh(tf.matmul(fc3,w4)+b4)
    return fc4, [w1,b1,w2,b2,w3,b3,w4,b4]

# re-used for optimizing all networks
def momentum_optimizer(loss,var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.003,                # Base learning rate.
        batch,  # Current index into the dataset.
        TRAIN_ITERS // 4,          # Decay step - this decays 4 times throughout training process.
        0.95,                # Decay rate.
        staircase=True)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer

# plot decision surface
def plot_d0(D,input_node,tmpax):
    
    ax = tmpax.add_subplot(111)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')
    # decision boundary
    r=1000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in a minibatch
    for i in range(r//M):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D,{input_node: x})

    ax.plot(xs, ds, label='decision boundary')
    ax.set_ylim(0,1.1)
    plt.legend()


def plot_fig(tmpax):
    # plots pg, pdata, decision boundary 
    ax=tmpax.add_subplot(111)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

    # decision boundary
    r=5000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in same minibatch
    for i in range(r//M):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D1,{x_node: x})

    ax.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs=np.linspace(-5,5,r)
    gs=np.zeros((r,1)) # generator function
    for i in range(r//M):
        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
        gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')

    # ylim, legend
    ax.set_ylim(0,1.1)
    plt.legend()
    

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Target distribution P_data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
#seed = 42
#np.random.seed(seed)
#tf.set_random_seed(seed)

mu,sigma=-1,1
xs=np.linspace(-5,5,1000)
plt.plot(xs, norm.pdf(xs,loc=mu,scale=sigma))
plt.show()
#plt.savefig('fig0.png')


TRAIN_ITERS=10000
M=200 # minibatch size
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pre train Decision Surface
If decider is reasonably accurate to start, we get much faster convergence.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    
with tf.variable_scope("D_pre"):
    input_node=tf.placeholder(tf.float32, shape=(M,1),name="input_node")
    train_labels=tf.placeholder(tf.float32,shape=(M,1),name="train_labels")
    D,theta=mlp(input_node,1)
    loss=tf.reduce_mean(tf.square(D-train_labels))

with tf.name_scope("D_pre_training"):
    optimizer=momentum_optimizer(loss,None)
    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    tmpax = plt.figure()
    plot_d0(D,input_node,tmpax)
    plt.title('Initial Decision Boundary')
    plt.show()
    #plt.savefig('fig1.png')
    lh=np.zeros(1000)
    for i in range(1000):
        #d=np.random.normal(mu,sigma,M)
        d=(np.random.random(M)-0.5) * 10.0 # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
        labels=norm.pdf(d,loc=mu,scale=sigma)
        lh[i],_=sess.run([loss,optimizer], {input_node: np.reshape(d,(M,1)), train_labels: np.reshape(labels,(M,1))})

# training loss
plt.figure()
plt.plot(lh)
plt.title('Training Loss')    
plt.show()

tmpax = plt.figure()
plot_d0(D,input_node,tmpax)
plt.show()
#plt.savefig('fig2.png')

# copy the learned weights over into a tmp array
weightsD=sess.run(theta)
# close the pre-training session
sess.close()

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Construct G and D NN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
with tf.name_scope("G_construct"):
    with tf.variable_scope("G"):
        z_node=tf.placeholder(tf.float32, shape=(M,1),name="z_node") # M uniform01 floats
        G,theta_g=generator(z_node,1) # generate normal transformation of Z
        G=tf.mul(5.0,G) # scale up by 5 to match range
with tf.name_scope("D_construct"):
    with tf.variable_scope("D") as scope:
        # D(x)
        x_node=tf.placeholder(tf.float32, shape=(M,1),name="x_node") # input M normally distributed floats
        fc,theta_d=mlp(x_node,1) # output likelihood of being normally distributed
        D1=tf.maximum(tf.minimum(fc,.99), 0.01,name="D1") # clamp as a probability
        # make a copy of D that uses the same variables, but takes in G as input
        scope.reuse_variables()
        fc,theta_d=mlp(G,1)
        D2=tf.maximum(tf.minimum(fc,.99), 0.01,name="D2")
 
with tf.name_scope("training"):
    obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2),name="obj_d")
    obj_g=tf.reduce_mean(tf.log(D2),name="obj_g")
    
    # set up optimizer for G,D
    opt_d=momentum_optimizer(-obj_d, theta_d)
    opt_g=momentum_optimizer(-obj_g, theta_g) # maximize log(D(G(z)))
    
    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    with tf.name_scope("Assign_Dpre_W"):
        # copy weights from pre-training over to new D network
        for i,v in enumerate(theta_d):
            sess.run(v.assign(weightsD[i]))
        
    # initial conditions
    tmpax = plt.figure()
    plot_fig(tmpax)
    plt.title('Before Training')
    #plt.savefig('fig3.png')
    plt.show()
    # Algorithm 1 of Goodfellow et al 2014
    
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Here: Training 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    
    k=1
    histd, histg= np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)
    for i in range(TRAIN_ITERS):
        for j in range(k):
            x= np.random.normal(mu,sigma,M) # sampled m-batch from p_data
            x.sort()
            z= np.linspace(-5.0,5.0,M)+np.random.random(M)*0.01  # sample m-batch from noise prior
            histd[i],_=sess.run([obj_d,opt_d], {x_node: np.reshape(x,(M,1)), z_node: np.reshape(z,(M,1))})
        z= np.linspace(-5.0,5.0,M)+np.random.random(M)*0.01 # sample noise prior
        histg[i],_=sess.run([obj_g,opt_g], {z_node: np.reshape(z,(M,1))}) # update generator
        if i % (TRAIN_ITERS//10) == 0:
            train_writer = tf.summary.FileWriter('./summaries/',sess.graph)
            print(float(i)/float(TRAIN_ITERS))

plt.figure()
plt.plot(range(TRAIN_ITERS),histd, label='obj_d')
plt.plot(range(TRAIN_ITERS), 1-histg, label='obj_g')
plt.legend()
#plt.savefig('fig4.png')

tmpax = plt.figure()
plot_fig(tmpax)
#plt.savefig('fig5.png')
# close the pre-training session
sess.close()
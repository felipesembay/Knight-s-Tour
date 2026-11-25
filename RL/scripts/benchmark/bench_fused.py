import os, sys, time, json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
dev=sys.argv[1]; N=int(sys.argv[2]); B=int(sys.argv[3])
if dev=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
def build():
    return Sequential([Conv2D(32,3,activation='relu',padding='same',input_shape=(N,N,3)),Conv2D(64,3,activation='relu',padding='same'),Flatten(),Dense(64,activation='relu'),Dense(8)])
q,qt=build(),build(); qt.set_weights(q.get_weights())
opt=tf.keras.optimizers.Adam(5e-4,clipnorm=1.0,jit_compile=False)
huber=tf.keras.losses.Huber()
@tf.function
def step(s,a,r,s2,d,m2):
    nq=q(s2,training=False); nq=tf.where(m2,nq,-1e9); best=tf.argmax(nq,1,output_type=tf.int32)
    tq=tf.gather(qt(s2,training=False),best,batch_dims=1)
    tgt=r+0.99*tq*(1.0-d)*tf.cast(tf.reduce_any(m2,1),tf.float32)
    with tf.GradientTape() as tape:
        qa=tf.gather(q(s,training=True),a,batch_dims=1); loss=huber(tgt,qa)
    opt.apply_gradients(zip(tape.gradient(loss,q.trainable_variables),q.trainable_variables)); return loss
s=np.random.rand(B,N,N,3).astype('f4'); a=np.random.randint(0,8,B).astype('i4'); r=np.random.rand(B).astype('f4'); d=np.zeros(B,'f4'); m=np.random.rand(B,8)>0.5
for _ in range(10): step(s,a,r,s,d,m)
t=time.perf_counter()
for _ in range(200): step(s,a,r,s,d,m)
print(json.dumps({'device':dev,'board':N,'batch':B,'fused_step_ms':round((time.perf_counter()-t)/200*1000,3)}))

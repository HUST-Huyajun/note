# Ipython(jupyter notebook) 动态图更新

```python
from IPython import display
import matplotlib.pyplot as plt
# plot the real data using plt old version
plt.scatter(x_data,y_data)
plt.ion()
plt.show()
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(3000):
        sess.run(trainstep,feed_dict={xs:x_data,ys:y_data})
        if i % 50==0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            predict=sess.run(prediction,feed_dict={xs:x_data})
            # plot the prediction
            plt.cla()
            plt.scatter(x_data, y_data)
            plt.plot(x_data, predict, 'r-', lw=1)
            #plt.scatter(x_data,predict,c='red')
            plt.pause(0.15)
            #clear Ipython
            display.clear_output(wait=True)
```
# python 动态图更新

```python
# plot the real data
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(3000):
        sess.run(trainstep,feed_dict={xs:x_data,ys:y_data})
        if i % 50==0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            predict=sess.run(prediction,feed_dict={xs:x_data})
            # plot the prediction
            lines = ax.plot(x_data, predict, 'r-', lw=1)
            plt.pause(0.15)
```
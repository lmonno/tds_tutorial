
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

eps = np.random.normal(loc = 0 , scale =5, size = 1000)
x_seq = np.arange(1,11,1).reshape(-1,1)
x = np.resize(x_seq, 1000)
a = 1
b = -30
c = -3
y = a * np.power(x, 2.5) + b * x + c +eps


x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


degree=3
polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
polyreg.fit(x,y)

y_pred = polyreg.predict(x_seq)#.reshape(-1,1)
fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=np.squeeze(x), y=np.squeeze(y),
                    mode='markers',
                    name='markers'))
fig.add_trace(go.Scatter(x=np.squeeze(x_seq), y=np.squeeze(y_pred),
                    mode='lines',
                    name='lines'))

fig.show()
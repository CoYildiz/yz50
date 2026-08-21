import math
import matplotlib.pyplot as plt

inputs = [1,2,3,4,5]
weights = [[1,2,3,0.5,-2],[2,-3,4,-5,3],[1,-2.3,4.2,-1.9,0.2],[0.1,0.2,0.3,0.5,0.1],[2.3,-1.4,7.1,4.1,3]]
weights_basic = [1,2,2,-2,-1]
bias = 3
expected_values = [1,0,0.3,0.2,0.4]

def sigmoid(z):
    return 1/(1 + math.exp(-z) )

def forward_pass_basic(inputs,weights,bias = 0):
    sumofiw = 0
    for i, w in zip(inputs,weights):
        sumofiw += i * w
    sumofiw += bias

    return sigmoid(sumofiw)


print(forward_pass_basic(inputs,weights_basic,bias))
# 247 pass

def forward_pass(inputs,weights,bias = 0):
    norons = []
    for weight in weights:
        noron = forward_pass_basic(inputs,weight,bias)
        norons.append(noron)
            
    return norons
print(forward_pass(inputs,weights,bias))

def loss_func(norons:list[int]):
    loss = 0
    for n, e in zip(norons, expected_values):
        loss += math.pow(n - e,2)
    return loss

print(loss_func(forward_pass(inputs,weights,bias)))


# for matplotlib
def matplotlib_draw():
    x_axis = []
    y_axis = []
    
    for w in [-2, -1, 0, 1, 2]:
        weights[2][0] = w
        
        current_norons = forward_pass(inputs, weights, bias)
        current_loss = loss_func(current_norons)
        
        x_axis.append(w)
        y_axis.append(current_loss)
        
    plt.plot(x_axis, y_axis)
    plt.xlabel("w[2][0]")
    plt.ylabel("loss func")
    plt.show()


def optimize_weight():
    h = 0.0001
    learning_rate = 1
    
    for i in range(100):
        loss_simdi = loss_func(forward_pass(inputs, weights, bias))
        
        weights[2][0] += h
        loss_nudge = loss_func(forward_pass(inputs, weights, bias))
        
        weights[2][0] -= h
        
        turev = (loss_nudge - loss_simdi) / h
        
        weights[2][0] -= learning_rate * turev
        
        print(f"Tur {i+1} - Loss: {loss_simdi}")

optimize_weight()
matplotlib_draw()

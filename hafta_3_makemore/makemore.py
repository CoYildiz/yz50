import torch
import matplotlib.pyplot as plt

with open("names.txt") as f:
    lines = f.readlines()
    words = [line.strip() for line in lines]
    #print(words[:10])
# === for learning and controlling stuff
# print(len(words))
# print(min(len(word) for word in words))
# print(max(len(word) for word in words))
# b = {}
# for w in words: # [e] # [m,m,a]
#     chs = ["<S>"] + list(w) + ["<E>"]
#     for ch1, ch2 in zip(chs, chs[1:]):
#         b[(ch1, ch2)] = b.get((ch1, ch2), 0) + 1

# ==============
# a to a matrix sum up
# N[0][0] = 556 it is same code below
# for key, value in b.items():
#     if key[0] == "a" and key[1] == "a":
#         print(key, value)
# ==================
N = torch.zeros((27, 27), dtype = torch.int32)
# we converted 27x27 matrix because we are wasting memory for example:
# <E> char is not used at the end of the word and too more
chars = sorted(list(set("".join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
# before this change we implemented that:
# stoi["<S>"] = 26
# stoi["<E>"] = 27
# but we changed it to because explanation above:
stoi["."] = 0



for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
#print(N)
# examples
# N[0][0] = 556
# a[0][0] = 2.2
# a[0][1] = 3
# out = a[0][0] + a[0][1] * 5
# print(int(out))
# for i in range(int(out)):
#     print(i)




## for visuliation ??
itos = {i:s for s,i in stoi.items()}
plt.figure(figsize=(16,16))
plt.imshow(N, cmap = "Purples")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="black")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.savefig("output/N_matrix.png")


# first row probability
# p = N[0].float()
# p = p / p.sum()
# print(p)

# g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator=g)
# p = p / p.sum()
# # print(p)
# torch.multinomial(p, num_samples=20, replacement=True, generator=g)

# we added 1 to N to avoid zero probabilities
P = (N+1).float()
P /= P.sum(1, keepdim = True)

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    ix = 0
    out = []
    while True:
        p = P[ix]
        # p = N[ix].float()
        # p = p / p.sum()
        # # p = torch.ones(27) / 27.0
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))


# log(a*b*c) = log(a) + log(b) + log(c
log_likelihood = 0.0
n = 0
for w in words[0:3]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        # print(f"{ch1}{ch2} -> {prob:.4f}")
        # print(f"{ch1}{ch2} -> {log_prob:.4f}")
        n += 1
print(f"log_likelihood: {log_likelihood}")
# negative log likelihood because we are maximizing log likelihood
nll = -log_likelihood
print(f"nll: {nll}")
print(f"average nnl: {nll / n}")


# create the training set of bigrams

xs, ys = [], []
for w in words[:1]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(f"{ch1}{ch2}")
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs, ys)

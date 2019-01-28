import pickle
a = [1,2,3]
file = open('test.pkl', 'wb+')
pickle.dump(a, file)
file.close()
file = open('test.pkl', 'rb')
b = pickle.load(file)
print(b)

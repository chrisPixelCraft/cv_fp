fixframe = range(5)
state = [True for i in range(len(fixframe))] 
state = [False, True, True, False, True]

start = 0
end = 0
deleted = set() # id in fixframe
while end < len(fixframe) and end < len(fixframe):
    while state[end] == state[start]:
        end += 1
        if end == len(state):
            break
    candidate = state[start:end]
    distances = range(len(candidate))
    max_distance = max(distances)
    for i in range(len(candidate)):
        if distances[i] < max_distance:
            deleted.add(start+i)
    start = end
    end = start + 1

print([fixframe[i] for i in range(len(fixframe)) if i not in deleted])


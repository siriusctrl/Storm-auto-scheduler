nums = ['abcd', 'dce', 'abs']

def find(nums):
    firstmap = {}

    for n in nums:
        if n[0] in firstmap:
            firstmap[n[0]] = firstmap[n[0]] + [n]
        else:
             firstmap[n[0]] = [n]

    res = []

    for n in nums:
        if n[-1] in firstmap:
            for i in firstmap[n[-1]]:
                res.append((n,i))
    
    return res
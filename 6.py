#P[0][1] = P(1|HOT)
#P[1][3] = P(3|COLD)
#...
P = [[0, 0.2, 0.4, 0.4],
     [0, 0.5, 0.4, 0.1]]

#AA, AB, BA, BB
#movement[0][0]: move prob from A to A
#movement[1][0]: move prob from B to A
#...
movement = [[0.6, 0.3],
            [0.4, 0.5]]

#move prob from Start to A, B
StartToAB = [0.8,0.2]

#end prob
endProb = 0.1

def prob(sequence, sourceList):
    res = 1
    res*= StartToAB[sourceList[0]] * P[sourceList[0]][sequence[0]]
    # print(StartToAB[sourceList[0]] , P[sourceList[0]][sequence[0]]),
    for i in xrange(1,4):
        res*= movement[sourceList[i-1]][sourceList[i]] * P[sourceList[i]][sequence[i]]
        # print(movement[sourceList[i-1]][sourceList[i]] , P[sourceList[i]][sequence[i]]),
    # print("\n")
    res *= endProb
    return res

def sequenceProb(sequence):
    res = 0
    for a in xrange(0,2):
        for b in xrange(0,2):
            for c in xrange(0,2):
                for d in xrange(0,2):
                    # print("Calc prob for ", a,b,c,d)
                    res += prob(sequence, [a,b,c,d])

    return res

print(sequenceProb([3,2,1,3]))
print(sequenceProb([3,2,1,2]))